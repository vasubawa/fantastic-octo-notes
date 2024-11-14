import os
import markdown2
import pdfkit

def format_notes(summary_file_name, pdf_file_name):
    # Read the summarized text
    with open(summary_file_name, "r", encoding="utf-8") as summary_file:
        summarized_text = summary_file.read()

    # Convert Markdown to HTML
    html_content = markdown2.markdown(summarized_text)

    # Write the HTML content to a temporary .html file
    html_file_name = os.path.join('createdFiles', os.path.basename(pdf_file_name).replace('.pdf', '.html'))
    with open(html_file_name, "w", encoding="utf-8") as html_file:
        html_file.write(html_content)

    # Convert the .html file to a PDF using pdfkit
    try:
        pdfkit.from_file(html_file_name, pdf_file_name)
    except Exception as e:
        print(f"Error during PDF generation: {e}")

    # Clean up the .html file
    if os.path.exists(html_file_name):
        os.remove(html_file_name)
