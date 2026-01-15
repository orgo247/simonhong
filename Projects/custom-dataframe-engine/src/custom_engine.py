def parser(filepath):
    data=[]
    with open (filepath, 'r') as file:
        for line in file:
            row = line.rstrip("\n\r")
            if not row:
                continue
            row = row.split(",")
            data.append(row)
    return data


class DataFrame:
    def __init__(self, filepath=None, header=None, data=None):
        if filepath is not None:
            file = parser(filepath)
            if not file or len(file) <2:
                raise ValueError("File is empty or cannot be read")
            self.header = list(file[0])
            self.rows= file[1:]
        else:
            self.header = list(header) if header else []
            if header and data:
                self.rows = list(zip(*[data[c] for c in header])) if header and data else []
            else:
                self.rows = []

        self.data = {col: [row[i] for row in self.rows] for i, col in enumerate(self.header)}
        self.nrows = len(self.rows)

    @classmethod
    def from_original(cls, header, data):
        return cls(header=header, data=data)


    def head(self, n=5):
        n = max(0, min(n, self.nrows))
        new_data = {col: self.data[col][:n] for col in self.header}
        return DataFrame.from_original(header=self.header, data=new_data)
    # bracket
    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self.header:
                raise KeyError(f"Column {key} does not exist")
            return Series(self.data[key])
        if isinstance(key, list) and all(isinstance(k, str) for k in key):
            for col in key:
                if col not in self.header:
                    raise KeyError(f"Column {col} does not exist")
            new_data = {col: self.data[col][:] for col in key}
            return DataFrame.from_original(header=key, data=new_data)
        if isinstance(key, list) and all(isinstance(k, bool) for k in key):
            if len(key) != self.nrows:
                raise ValueError("Boolean index length does not match number of rows")
            indices = [i for i, flag in enumerate(key) if flag]
            new_data = {col: [self.data[col][i] for i in indices] for col in self.header}
            return DataFrame.from_original(header=self.header, data=new_data)
        if isinstance(key, slice):
            start, stop, step = key.indices(self.nrows)
            indices = range(start, stop, step)
            new_data = {col: [self.data[col][i] for i in indices] for col in self.header}
            return DataFrame.from_original(header=self.header, data=new_data)

    def __repr__(self):
        if not self.header:
            return "0×0 []"

        lines = []
        lines.append(" | ".join(self.header))
        lines.append("-" * (len(self.header) * 10))
        for i in range(self.nrows):
            lines.append(" | ".join(str(self.data[c][i]) for c in self.header))
        return "\n".join(lines)

    def drop(self, cols):
        if isinstance(cols, str):
            cols = [cols]

        for c in cols:
            if c not in self.header:
                raise KeyError(f"Column {c} does not exist")

        keep_cols = [c for c in self.header if c not in cols]
        new_data = {col: self.data[col][:] for col in keep_cols}

        return DataFrame.from_original(header=keep_cols, data=new_data)

    def __delitem__(self, key):
        if key not in self.header:
            raise KeyError(f"Column {key} does not exist")

        del self.data[key]
        self.header.remove(key)


    # projection
    def select(self, cols):
        if isinstance(cols, str):
            cols = [cols]
        for c in cols:
            if c not in self.header:
                raise KeyError(f" column {c} does not exist")
        new_data = {col: self.data[col][:] for col in cols}
        return DataFrame.from_original(header=cols, data=new_data)

    # filtering
    def filter(self, column, operator, value):
        if column not in self.header:
            raise KeyError(f"Column {column} does not exist")
        if operator not in ['==', '!=', '<', '<=', '>', '>=']:
            raise ValueError(f"Operator {operator} is not supported")
        column_data = self.data[column]

        def try_num(x):
            if isinstance(x, (int, float)):
                return x
            try:
                if "." in str(x):
                    return float(x)
                return int(x)
            except Exception:
                return x

        value = try_num(value)

        def condition(input):
            input = try_num(input)
            if operator == "==":
                return input == value
            elif operator == "!=":
                return input != value
            elif operator == "<":
                return input < value
            elif operator == "<=":
                return input <= value
            elif operator == ">":
                return input > value
            elif operator == ">=":
                return input >= value

        indices = [i for i,input in enumerate(column_data) if condition(input)]
        new_data = {col: [self.data[col][i] for i in indices] for col in self.header}
        return DataFrame.from_original(header=self.header, data=new_data)
    # groupby
    def groupby(self, keys):
        for k in keys:
            if k not in self.header:
                raise KeyError(f"Column {k} does not exist")
        return Groupby_(self, keys)
    # join
    def join(self, other, left_on, right_on, how="inner"):
        if how!="inner":
            raise ValueError(f"Join type {how} is not supported")
        if left_on not in self.header:
            raise KeyError(f"Column {left_on} does not exist in left DataFrame")
        if right_on not in other.header:
            raise KeyError(f"Column {right_on} does not exist in right DataFrame")

        index= {}
        for i in range(other.nrows):
            key = other.data[right_on][i]
            index.setdefault(key, []).append(i)

        result_header = list(self.header)
        right_data = []
        for col in other.header:
            if col == right_on:
                continue
            result_column = col if col not in result_header else col + "_right"
            result_header.append(result_column)
            right_data.append( (col, result_column) )
        result_data = {col: [] for col in result_header}

        for i in range(self.nrows):
            key = self.data[left_on][i]
            matching_indices = index.get(key)
            if not matching_indices:
                continue
            for j in matching_indices:
                for col in self.header:
                    result_data[col].append(self.data[col][i])
                for (col, result_column) in right_data:
                    result_data[result_column].append(other.data[col][j])

        return DataFrame.from_original(header=result_header, data=result_data)



    def __setitem__(self, key, value):
        if not isinstance(value, list)and not isinstance(value, Series):
            value = [value] * self.nrows
        if len(value) != self.nrows:
            raise ValueError("Length of new column does not match number of rows")
        self.data[key] = value
        if key not in self.header:
            self.header.append(key)

    def with_column(self, column, values):
        if len(values) != self.nrows:
            raise ValueError("Length of new column does not match number of rows")
        new_header = list(self.header)
        if column not in new_header:
            new_header.append(column)
        new_data = {col: self.data[col][:] for col in self.header}
        new_data[column] = list(values)
        return DataFrame.from_original(header=new_header, data=new_data)

    def __len__(self):
        return self.nrows


class Groupby_:
    def __init__(self, df, keys):
        self.df = df
        self.keys = keys

    def agg(self, option):
        df, keys = self.df, self.keys
        groups = {}
        for i in range(df.nrows):
            key_tuple = tuple(df.data[col][i] for col in keys)
            groups.setdefault(key_tuple, []).append(i)

        result_cols = list(keys)
        result_data = {c: [] for c in result_cols}

        def result_name(col, function):
            return f"{col}_{function}"

        for col, f in option.items():
            result_cols.append(result_name(col, f))
            result_data[result_name(col, f)] = []

        for key_tuple,indices in groups.items():
            for i, key_col in enumerate(keys):
                result_data[key_col].append(key_tuple[i])
            for col, function in option.items():
                values = [df.data[col][i] for i in indices]
                if function == "count":
                    result_data[result_name(col, function)].append(len(indices))
                    continue

                nums = []
                for value in values:
                    try:
                        nums.append(float(value))
                    except Exception:
                        pass
                if function == "sum":
                    result_data[result_name(col, function)].append(sum(nums))
                elif function == "avg":
                    result_data[result_name(col, function)].append(round(sum(nums)/len(nums),4) if len(nums)>0 else None)
                elif function == "min":
                    result_data[result_name(col, function)].append(min(nums) if nums else None)
                elif function == "max":
                    result_data[result_name(col, function)].append(max(nums) if nums else None)
                else:
                    raise ValueError(f"Function {function} is not supported")
        return DataFrame.from_original(header=result_cols, data=result_data)


class Series(list):
    def __eq__(self, other):
        try:
            return [float(x) == float(other) for x in self]
        except (ValueError, TypeError):
            # Fallback to string comparison if not numbers
            return [x == other for x in self]

    def __lt__(self, other):
        return [float(x) < float(other) for x in self]

    def __gt__(self, other):
        return [float(x) > float(other) for x in self]

    def __le__(self, other):
        return [float(x) <= float(other) for x in self]

    def __ge__(self, other):
        return [float(x) >= float(other) for x in self]

    def __ne__(self, other):
        try:
            return Series([float(x) != float(other) for x in self])
        except (ValueError, TypeError):
            return Series([x != other for x in self])
    def __mul__(self, other):
        try:
            return Series([round(float(x) * float(other),4) for x in self])
        except (ValueError, TypeError):
            return Series([x for x in self])
    def __add__(self, other):
        return Series([float(x) + float(other) for x in self])
    def __sub__(self, other):
        return Series([float(x) - float(other) for x in self])
    def __truediv__(self, other):
        return Series([round(float(x) / float(other),4) for x in self])