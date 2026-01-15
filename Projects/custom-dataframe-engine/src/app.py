import streamlit as st, sys, io, textwrap
from custom_engine import DataFrame

st.set_page_config(page_title="custom-engine", layout="wide")

if "ns" not in st.session_state:
    st.session_state.ns = {"DataFrame": DataFrame, "len": len}

if "cells" not in st.session_state:
    st.session_state.cells = [{
        "code": "",
        "out": "",
        "err": ""
    }]

ns = st.session_state.ns
cells = st.session_state.cells

def list_loaded(ns):
    out = {}
    for name, obj in ns.items():
        if isinstance(obj, DataFrame):
            out[name] = (obj.nrows, len(obj.header))
    return out


def exec_cell(code: str):
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        src = code.strip()

        try:
            compiled = compile(src, "<cell>", "eval")
            result = eval(compiled, {}, ns)
            if result is not None:
                print(result)
        except SyntaxError:
            exec(src, {}, ns)

        out = buf.getvalue().rstrip()
        return (out if out else "", "")
    except Exception as e:
        return ("", f"{type(e).__name__}: {e}")
    finally:
        sys.stdout = old

st.sidebar.header("Loaded DataFrames")
loaded = list_loaded(ns)
if not loaded:
    st.sidebar.caption('Empty')
else:
    for name, (r, c) in loaded.items():
        st.sidebar.write(f"**{name}** — {r} rows × {c} cols")

if st.sidebar.button("Clear session"):
    st.session_state.ns = {"DataFrame": DataFrame, "len": len}
    st.session_state.cells = [{
        "code": "# session cleared",
        "out": "",
        "err": ""
    }]
    st.rerun()

st.title("custom-engine")

for i, cell in enumerate(cells):
    with st.container(border=True):
        st.markdown(f"**Cell {i+1}**")
        code_val = st.text_area("Code", value=cell["code"], key=f"code_{i}", height=140)

        c1, c2, _ = st.columns([1,1,6])
        run = c1.button("Run", key=f"run_{i}")
        del_ = c2.button("Delete", key=f"del_{i}")

        if run:
            out, err = exec_cell(code_val)
            cells[i]["code"] = code_val
            cells[i]["out"] = out
            cells[i]["err"] = err
            st.rerun()

        if del_:
            cells.pop(i)
            if not cells:
                cells.append({"code": "", "out": "", "err": ""})
            st.rerun()

        if cell["err"]:
            st.error(cell["err"])
        if cell["out"]:
            st.code(cell["out"], language="text")

st.divider()
if st.button("➕ Add new cell"):
    cells.append({"code": "", "out": "", "err": ""})
    st.rerun()