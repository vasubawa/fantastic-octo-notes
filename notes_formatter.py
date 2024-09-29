import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem

def format_notes(summary_file_name, pdf_file_name):
    # Create a PDF document
    doc = SimpleDocTemplate(pdf_file_name, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(name='Title', fontSize=18, leading=22, spaceAfter=12, alignment=1)
    heading_style = ParagraphStyle(name='Heading2', fontSize=14, leading=18, spaceAfter=12)
    bullet_style = ParagraphStyle(name='Bullet', parent=styles['BodyText'], bulletIndent=20, spaceAfter=6)

    with open(summary_file_name, "r", encoding="utf-8") as summary_file:
        summary_text = summary_file.read()

    # Add title
    story.append(Paragraph("Notes", title_style))
    story.append(Spacer(1, 12))

    # Split the summary into paragraphs
    paragraphs = summary_text.split("\n\n")
    
    for paragraph in paragraphs:
        # Detect section headings (typically end with a colon in your case)
        if re.match(r".+:", paragraph):
            story.append(Paragraph(paragraph.strip(), heading_style))
            story.append(Spacer(1, 12))
        else:
            # If not a heading, turn sentences into bullet points
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)
            bullet_points = []
            for sentence in sentences:
                if ':' in sentence:
                    key, value = sentence.split(':', 1)
                    bullet_points.append(Paragraph(f"**{key.strip()}:** {value.strip()}", bullet_style))
                else:
                    bullet_points.append(Paragraph(sentence.strip(), bullet_style))
            story.append(ListFlowable(bullet_points, bulletType='bullet', start='circle'))
            story.append(Spacer(1, 12))

    # Build the PDF
    doc.build(story)
    print(f"Formatted notes saved to {pdf_file_name}")

def main():
    summary_file_name = "createdFiles/summary.txt"
    pdf_file_name = "createdFiles/notes.pdf"
    format_notes(summary_file_name, pdf_file_name)

if __name__ == "__main__":
    main()