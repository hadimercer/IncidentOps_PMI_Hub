from pathlib import Path

path = Path(r'C:\Users\belgh\Projects\hadimercer-portfolio\incidentops-pmi-hub\app.py')
text = path.read_text()
imports = "from ui.charts import annotate as annotate_chart, style_figure\nfrom ui.primitives import (\n    empty_state,\n    export_frame,\n    global_filter_bar,\n    insight_callout,\n    metric_strip,\n    narrative_callouts,\n    page_masthead,\n    section_header,\n    sidebar_brand,\n    summary_table,\n    worklist_table,\n)\nfrom ui.theme import TOKENS, apply_theme\n"
if 'from ui.charts import annotate as annotate_chart, style_figure' not in text:
    anchor = 'from dotenv import load_dotenv\n'
    text = text.replace(anchor, anchor + imports)
marker = 'def main() -> None:\n'
head, sep, tail = text.partition(marker)
if not sep:
    raise SystemExit('main() marker not found')
new_block = Path(r'C:\Users\belgh\Projects\hadimercer-portfolio\incidentops-pmi-hub\tmp_main_block.txt').read_text()
path.write_text(head + new_block)
print('updated app.py')
