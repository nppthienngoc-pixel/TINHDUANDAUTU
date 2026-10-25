import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Thẩm định hiệu quả dự án đầu tư", page_icon="📈", layout="wide")

# ======================================================================
# CÁC HÀM TÍNH TOÁN
# ======================================================================


def npv(rate, cfs):
    """NPV = sum( CF_t / (1+r)^t ),  t = 0..n"""
    t = np.arange(len(cfs))
    return float(np.sum(np.asarray(cfs, dtype=float) / (1 + rate) ** t))


def irr(cfs):
    """Tìm các nghiệm r sao cho NPV(r) = 0 (quét lưới + chia đôi)."""
    cfs = np.asarray(cfs, dtype=float)
    if not (np.any(cfs < 0) and np.any(cfs > 0)):
        return []
    t = np.arange(len(cfs))
    f = lambda r: float(np.sum(cfs / (1 + r) ** t))

    grid = np.linspace(-0.99, 10, 4000)
    vals = np.array([f(r) for r in grid])
    roots = []
    for i in range(len(grid) - 1):
        a, b, fa, fb = grid[i], grid[i + 1], vals[i], vals[i + 1]
        if fa == 0:
            roots.append(a)
        elif fa * fb < 0:
            for _ in range(100):
                m = (a + b) / 2
                fm = f(m)
                if fa * fm <= 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            roots.append((a + b) / 2)
    return roots


def mirr(cfs, finance_rate, reinvest_rate):
    """MIRR = (FV dòng tiền dương tại lãi suất tái đầu tư / |PV dòng tiền âm tại lãi suất tài trợ|)^(1/n) - 1"""
    cfs = np.asarray(cfs, dtype=float)
    n = len(cfs) - 1
    t = np.arange(len(cfs))
    pos = np.where(cfs > 0, cfs, 0.0)
    neg = np.where(cfs < 0, cfs, 0.0)
    fv_pos = float(np.sum(pos * (1 + reinvest_rate) ** (n - t)))
    pv_neg = float(np.sum(neg / (1 + finance_rate) ** t))
    if fv_pos <= 0 or pv_neg == 0 or n <= 0:
        return None, fv_pos, pv_neg
    return (fv_pos / abs(pv_neg)) ** (1 / n) - 1, fv_pos, pv_neg


def bcr(rate, benefits, costs):
    """BCR = PV(Lợi ích) / PV(Chi phí)"""
    t = np.arange(len(benefits))
    pv_b = float(np.sum(np.asarray(benefits, dtype=float) / (1 + rate) ** t))
    pv_c = float(np.sum(np.asarray(costs, dtype=float) / (1 + rate) ** t))
    return (pv_b / pv_c if pv_c != 0 else None), pv_b, pv_c


def payback(cfs):
    """Thời gian hoàn vốn (nội suy tuyến tính trong năm hoàn vốn)."""
    cum = np.cumsum(cfs)
    if cum[0] >= 0:
        return 0.0, cum
    for i in range(1, len(cfs)):
        if cum[i] >= 0:
            if cfs[i] == 0:
                return float(i), cum
            return (i - 1) + (-cum[i - 1]) / cfs[i], cum
    return None, cum


def fmt(x, d=2):
    return f"{x:,.{d}f}"


def years_months(val):
    return f"≈ {int(val)} năm {round((val - int(val)) * 12)} tháng"


# ======================================================================
# GIAO DIỆN
# ======================================================================
st.title("📈 Thẩm định hiệu quả dự án đầu tư")
st.caption("Bước 1: Nhập chỉ tiêu dự án → Bước 2: Chọn các chỉ tiêu cần tính → Xem kết quả.")

# ----------------------------------------------------------------------
# BƯỚC 1: NHẬP DỮ LIỆU DỰ ÁN
# ----------------------------------------------------------------------
st.header("Bước 1. Nhập dữ liệu dự án")

c1, c2, c3, c4 = st.columns(4)
with c1:
    n = int(st.number_input("Số năm hoạt động (n)", min_value=1, max_value=50, value=5, step=1))
with c2:
    r = st.number_input("Suất chiết khấu r (%/năm)", value=10.0, step=0.5, format="%.2f") / 100
with c3:
    fin = st.number_input("Lãi suất tài trợ (%/năm) – cho MIRR", value=8.0, step=0.5, format="%.2f") / 100
with c4:
    reinv = st.number_input("Lãi suất tái đầu tư (%/năm) – cho MIRR", value=10.0, step=0.5, format="%.2f") / 100

st.write(
    "Nhập dòng tiền từng năm (năm 0 là vốn đầu tư ban đầu).  \n"
    "• **Dòng tiền ròng CF**: dùng cho NPV, IRR, MIRR, PP, DPP (vốn đầu tư nhập **số âm**).  \n"
    "• **Lợi ích B** và **Chi phí C**: chỉ cần nhập nếu muốn tính **BCR**."
)

df0 = pd.DataFrame({
    "Năm": list(range(n + 1)),
    "Dòng tiền ròng CF": 0.0,
    "Lợi ích B": 0.0,
    "Chi phí C": 0.0,
})
df_in = st.data_editor(df0, hide_index=True, disabled=["Năm"], use_container_width=True, key=f"table_{n}")

cfs = df_in["Dòng tiền ròng CF"].fillna(0).to_numpy(dtype=float)
B = df_in["Lợi ích B"].fillna(0).to_numpy(dtype=float)
C = df_in["Chi phí C"].fillna(0).to_numpy(dtype=float)
t = np.arange(n + 1)

# ----------------------------------------------------------------------
# BƯỚC 2: CHỌN CHỈ TIÊU TÍNH TOÁN
# ----------------------------------------------------------------------
st.header("Bước 2. Chọn chỉ tiêu cần tính")

OPTIONS = {
    "NPV": "NPV – Giá trị hiện tại ròng",
    "IRR": "IRR – Tỷ suất hoàn vốn nội bộ",
    "MIRR": "MIRR – IRR hiệu chỉnh",
    "BCR": "BCR – Tỷ số lợi ích / chi phí",
    "PP": "PP – Thời gian hoàn vốn",
    "DPP": "DPP – Thời gian hoàn vốn có chiết khấu",
}

selected = st.multiselect(
    "Chọn một hoặc nhiều chỉ tiêu:",
    options=list(OPTIONS.keys()),
    format_func=lambda k: OPTIONS[k],
    placeholder="Chọn chỉ tiêu...",
)

if not selected:
    st.info("👆 Hãy chọn ít nhất một chỉ tiêu để xem kết quả.")
    st.stop()

if not np.any(cfs != 0) and not (np.any(B != 0) or np.any(C != 0)):
    st.warning("Bạn chưa nhập dữ liệu dòng tiền. Hãy điền vào bảng ở Bước 1.")
    st.stop()

# ----------------------------------------------------------------------
# KẾT QUẢ
# ----------------------------------------------------------------------
st.header("Kết quả")

summary = []   # (chỉ tiêu, giá trị, kết luận)
details = {}   # hàm hiển thị chi tiết cho từng chỉ tiêu

has_signs = bool(np.any(cfs < 0) and np.any(cfs > 0))

for m in [k for k in OPTIONS if k in selected]:

    if m == "NPV":
        pv = cfs / (1 + r) ** t
        val = float(pv.sum())
        summary.append(("NPV", fmt(val), "✅ Chấp nhận (NPV > 0)" if val > 0 else "❌ Không chấp nhận (NPV ≤ 0)"))

        def show_npv(pv=pv):
            st.latex(r"NPV=\sum_{t=0}^{n}\frac{CF_t}{(1+r)^t}")
            st.dataframe(pd.DataFrame({"Năm": t, "CF": cfs, "Hệ số chiết khấu": 1 / (1 + r) ** t,
                                       "PV(CF)": pv, "PV tích lũy": np.cumsum(pv)}),
                         hide_index=True, use_container_width=True)
        details["NPV"] = show_npv

    elif m == "IRR":
        roots = irr(cfs)
        if not roots:
            summary.append(("IRR", "Không xác định", "⚠️ Dòng tiền phải có cả số âm và số dương"))
            details["IRR"] = lambda: st.error("Không tìm được IRR (dòng tiền phải đổi dấu ít nhất một lần).")
        else:
            val = roots[0]
            summary.append(("IRR", f"{val * 100:.4f} %",
                            f"✅ Chấp nhận (IRR > r = {r * 100:.2f}%)" if val > r
                            else f"❌ Không chấp nhận (IRR ≤ r = {r * 100:.2f}%)"))

            def show_irr(roots=roots):
                st.latex(r"\sum_{t=0}^{n}\frac{CF_t}{(1+IRR)^t}=0")
                if len(roots) > 1:
                    st.warning("Dòng tiền đổi dấu nhiều lần nên có nhiều nghiệm IRR: "
                               + ", ".join(f"{x * 100:.2f}%" for x in roots)
                               + ". Nên dùng NPV hoặc MIRR để ra quyết định.")
                else:
                    st.write(f"IRR là suất chiết khấu làm NPV = 0: **{roots[0] * 100:.4f}%**")
            details["IRR"] = show_irr

    elif m == "MIRR":
        val, fv_pos, pv_neg = mirr(cfs, fin, reinv)
        if val is None:
            summary.append(("MIRR", "Không xác định", "⚠️ Dòng tiền phải có cả số âm và số dương"))
        else:
            summary.append(("MIRR", f"{val * 100:.4f} %",
                            f"✅ Chấp nhận (MIRR > r = {r * 100:.2f}%)" if val > r
                            else f"❌ Không chấp nhận (MIRR ≤ r = {r * 100:.2f}%)"))

        def show_mirr(val=val, fv_pos=fv_pos, pv_neg=pv_neg):
            st.latex(r"MIRR=\left(\frac{FV(CF^{+})}{|PV(CF^{-})|}\right)^{1/n}-1")
            a, b = st.columns(2)
            a.metric("FV dòng tiền dương (lãi suất tái đầu tư)", fmt(fv_pos))
            b.metric("PV dòng tiền âm (lãi suất tài trợ)", fmt(pv_neg))
            if val is None:
                st.error("Không tính được MIRR (cần có dòng tiền dương và dòng tiền âm).")
        details["MIRR"] = show_mirr

    elif m == "BCR":
        ratio, pv_b, pv_c = bcr(r, B, C)
        if ratio is None:
            summary.append(("BCR", "Không xác định", "⚠️ Chưa nhập Lợi ích B / Chi phí C"))
        else:
            summary.append(("BCR", fmt(ratio, 4),
                            "✅ Chấp nhận (BCR > 1)" if ratio > 1 else "❌ Không chấp nhận (BCR ≤ 1)"))

        def show_bcr(ratio=ratio, pv_b=pv_b, pv_c=pv_c):
            st.latex(r"BCR=\frac{PV(B)}{PV(C)}")
            a, b = st.columns(2)
            a.metric("PV lợi ích", fmt(pv_b))
            b.metric("PV chi phí", fmt(pv_c))
            if ratio is None:
                st.error("Tổng chi phí hiện tại bằng 0, không thể tính BCR. Hãy nhập cột Lợi ích B và Chi phí C.")
            st.dataframe(pd.DataFrame({"Năm": t, "B": B, "C": C,
                                       "PV(B)": B / (1 + r) ** t, "PV(C)": C / (1 + r) ** t}),
                         hide_index=True, use_container_width=True)
        details["BCR"] = show_bcr

    elif m == "PP":
        val, cum = payback(cfs)
        if val is None:
            summary.append(("PP", "Chưa hoàn vốn", "❌ Không hoàn vốn trong thời gian phân tích"))
        else:
            summary.append(("PP", f"{val:.2f} năm", years_months(val)))

        def show_pp(val=val, cum=cum):
            st.latex(r"PP=(k-1)+\frac{\left|\sum_{t=0}^{k-1}CF_t\right|}{CF_k}")
            st.dataframe(pd.DataFrame({"Năm": t, "CF": cfs, "CF tích lũy": cum}),
                         hide_index=True, use_container_width=True)
            st.line_chart(pd.DataFrame({"CF tích lũy": cum}, index=t))
        details["PP"] = show_pp

    elif m == "DPP":
        pv = cfs / (1 + r) ** t
        val, cum = payback(pv)
        if val is None:
            summary.append(("DPP", "Chưa hoàn vốn", "❌ Không hoàn vốn (có chiết khấu) trong thời gian phân tích"))
        else:
            summary.append(("DPP", f"{val:.2f} năm", years_months(val)))

        def show_dpp(val=val, cum=cum, pv=pv):
            st.latex(r"DPP=(k-1)+\frac{\left|\sum_{t=0}^{k-1}\frac{CF_t}{(1+r)^t}\right|}{\frac{CF_k}{(1+r)^k}}")
            st.dataframe(pd.DataFrame({"Năm": t, "CF": cfs, "PV(CF)": pv, "PV tích lũy": cum}),
                         hide_index=True, use_container_width=True)
            st.line_chart(pd.DataFrame({"PV tích lũy": cum}, index=t))
        details["DPP"] = show_dpp

# Bảng tổng hợp
st.subheader("Tổng hợp")
st.dataframe(pd.DataFrame(summary, columns=["Chỉ tiêu", "Giá trị", "Nhận xét"]),
             hide_index=True, use_container_width=True)

# Chi tiết từng chỉ tiêu
st.subheader("Chi tiết")
for key, show in details.items():
    with st.expander(OPTIONS[key], expanded=False):
        show()
