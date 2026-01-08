import streamlit as st
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

st.set_page_config(page_title="Internal Marks Calculator", page_icon="🧮")
st.title("🎓 Internal Marks Calculator (Out of 40)")
st.write("Calculate internal marks with automatic conversion, pass prediction, and suggestions for upcoming CATs.")

# --- Conversion function ---
def convert_marks(cat1, cat2, cat3, assignment):
    cat1_conv = (cat1 / 50) * 12 if cat1 is not None else 0.0
    cat2_conv = (cat2 / 25) * 6 if cat2 is not None else 0.0
    cat3_conv = (cat3 / 50) * 12 if cat3 is not None else 0.0
    assign_conv = (assignment / 10) * 10 if assignment is not None else 0.0
    total = cat1_conv + cat2_conv + cat3_conv + assign_conv
    return cat1_conv, cat2_conv, cat3_conv, assign_conv, total

# Conversion helpers
def conv2raw_cat2(conv): return (conv / 6.0) * 25.0
def conv2raw_cat3(conv): return (conv / 12.0) * 50.0
def raw2conv_cat2(raw): return (raw / 25.0) * 6.0
def raw2conv_cat3(raw): return (raw / 50.0) * 12.0

# --- PDF generator ---
def generate_pdf(data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("🎓 Internal Marks Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    table_data = [["Subject", "CAT1 (50)", "CAT2 (25)", "CAT3 (50)", "Assign (10)", "Total (40)", "Status"]]
    for row in data:
        table_data.append([
            row["Subject"], row["CAT1 (50)"], row["CAT2 (25)"], row["CAT3 (50)"],
            row["Assignment (10)"], row["Total (out of 40)"], row["Status"]
        ])
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
    ]))
    elements.append(table)
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# --- App UI ---
num_subjects = st.number_input("Enter number of subjects", min_value=1, max_value=10, step=1)
results = []

for i in range(num_subjects):
    st.subheader(f"📘 Subject {i+1}")
    subject_name = st.text_input("Enter subject name", f"Subject {i+1}", key=f"subject_{i}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cat1 = st.number_input("CAT 1 (out of 50)", min_value=0.0, max_value=50.0, key=f"cat1_{i}")
    with col2:
        cat2 = st.number_input("CAT 2 (out of 25)", min_value=0.0, max_value=25.0, key=f"cat2_{i}")
    with col3:
        cat3 = st.number_input("CAT 3 (out of 50)", min_value=0.0, max_value=50.0, key=f"cat3_{i}")
    with col4:
        assignment = st.number_input("Assignment (out of 10)", min_value=0.0, max_value=10.0, key=f"assign_{i}")

    # Convert
    cat1_conv, cat2_conv, cat3_conv, assign_conv, total = convert_marks(cat1, cat2, cat3, assignment)

    # --- Pass prediction logic ---
    MAX_CAT2_CONV, MAX_CAT3_CONV, PASS_MARK = 6.0, 12.0, 24.0
    written_cat1, written_cat2, written_cat3 = cat1 > 0, cat2 > 0, cat3 > 0

    current_conv = (cat1_conv if written_cat1 else 0.0) + (cat2_conv if written_cat2 else 0.0) + (assign_conv if assignment > 0 else 0.0)
    rem_cat2_conv_max = 0.0 if written_cat2 else MAX_CAT2_CONV
    rem_cat3_conv_max = 0.0 if written_cat3 else MAX_CAT3_CONV
    possible_max_total = current_conv + rem_cat2_conv_max + rem_cat3_conv_max

    status, suggestion_text = "", ""

    if not written_cat1 and not written_cat2 and not written_cat3:
        status = f"⏺️ No CATs written yet. Minimum total required across CAT2 & CAT3 (plus CAT1) to reach {PASS_MARK} will depend on your CAT1."
    elif written_cat1 and not written_cat2 and not written_cat3:
        if possible_max_total < PASS_MARK:
            status = f"❌ Even with full marks in CAT2 and CAT3 you cannot reach {PASS_MARK}/40. Max possible = {possible_max_total:.2f}/40."
        else:
            needed_conv = PASS_MARK - (cat1_conv + assign_conv)
            total_rem_conv_capacity = rem_cat2_conv_max + rem_cat3_conv_max
            alloc_cat2_conv = needed_conv * (rem_cat2_conv_max / total_rem_conv_capacity)
            alloc_cat3_conv = needed_conv - alloc_cat2_conv
            alloc_cat2_conv = min(alloc_cat2_conv, rem_cat2_conv_max)
            alloc_cat3_conv = min(alloc_cat3_conv, rem_cat3_conv_max)
            needed_cat2_raw = conv2raw_cat2(max(0.0, alloc_cat2_conv))
            needed_cat3_raw = conv2raw_cat3(max(0.0, alloc_cat3_conv))
            status = "🟡 You’ve written only CAT1."
            suggestion_text = (
                f"To pass (24/40) you need approximately:\n"
                f"• CAT2: **{needed_cat2_raw:.2f} / 25**\n"
                f"• CAT3: **{needed_cat3_raw:.2f} / 50**"
            )

    elif written_cat1 and written_cat2 and not written_cat3:
        if possible_max_total < PASS_MARK:
            status = f"❌ Even with full marks in CAT3 you cannot reach {PASS_MARK}/40. Max possible = {possible_max_total:.2f}/40."
        else:
            needed_conv = PASS_MARK - (cat1_conv + cat2_conv + assign_conv)
            if needed_conv <= 0:
                status = "✅ Already secured 24 marks! CAT3 optional for pass."
            else:
                needed_cat3_raw = conv2raw_cat3(needed_conv)
                if needed_cat3_raw > 50.0:
                    status = f"❌ Needed CAT3 marks ({needed_cat3_raw:.2f}) exceed 50; not possible."
                else:
                    status = "🟡 You’ve written CAT1 & CAT2."
                    suggestion_text = f"To pass (24/40) you need about **{needed_cat3_raw:.2f} / 50** in CAT3."

    else:
        if total >= PASS_MARK:
            status = f"✅ PASS (Total: {total:.2f} / 40)"
        else:
            if possible_max_total < PASS_MARK:
                status = f"❌ Even with full marks in remaining CATs you cannot reach {PASS_MARK}/40. Max possible = {possible_max_total:.2f}/40."
            else:
                needed_conv = PASS_MARK - (cat1_conv + cat2_conv + cat3_conv + assign_conv)
                rem_caps = []
                if not written_cat2: rem_caps.append(('cat2', rem_cat2_conv_max))
                if not written_cat3: rem_caps.append(('cat3', rem_cat3_conv_max))
                total_capacity = sum(c for _, c in rem_caps)
                allocs = {}
                for name, cap in rem_caps:
                    alloc = needed_conv * (cap / total_capacity)
                    allocs[name] = min(cap, max(0.0, alloc))
                parts = []
                if 'cat2' in allocs:
                    parts.append(f"CAT2: {conv2raw_cat2(allocs['cat2']):.2f} / 25")
                if 'cat3' in allocs:
                    parts.append(f"CAT3: {conv2raw_cat3(allocs['cat3']):.2f} / 50")
                status = "🟡 Prediction:"
                suggestion_text = "To pass (24/40), you need approximately: " + " and ".join(parts)

    # --- Show Results (no expander) ---
    st.markdown(f"**Converted Marks for {subject_name}:**")
    st.write(f"- CAT 1: {cat1_conv:.2f} / 12")
    st.write(f"- CAT 2: {cat2_conv:.2f} / 6")
    st.write(f"- CAT 3: {cat3_conv:.2f} / 12")
    st.write(f"- Assignment: {assign_conv:.2f} / 10")
    if suggestion_text:
        st.markdown(f"**{status}**  \n{suggestion_text}")
    else:
        st.markdown(f"**{status}**")

    st.divider()

    results.append({
        "Subject": subject_name,
        "CAT1 (50)": cat1,
        "CAT2 (25)": cat2,
        "CAT3 (50)": cat3,
        "Assignment (10)": assignment,
        "Total (out of 40)": round(total, 2),
        "Status": "Pass" if total >= 24 else "Fail"
    })

# --- Summary + Downloads ---
if results:
    st.subheader("📊 Summary of All Subjects")
    df = pd.DataFrame(results)
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results as CSV", data=csv, file_name="internal_marks_results.csv", mime="text/csv")
    pdf_data = generate_pdf(results)
    st.download_button("📄 Download Results as PDF", data=pdf_data, file_name="internal_marks_report.pdf", mime="application/pdf")