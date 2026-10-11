"""Render every page and generate contact sheets plus text/layout diagnostics."""
from pathlib import Path
import json, re
import pymupdf as fitz
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[2]
pdf=ROOT/'output/pdf/rst19-starfield-paper.pdf'
out=ROOT/'tmp/paper-qa';out.mkdir(parents=True,exist_ok=True)
doc=fitz.open(pdf);thumbs=[];pages=[]
for i,p in enumerate(doc):
    p.get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False).save(out/f'page-{i+1:02}.png')
    im=Image.open(out/f'page-{i+1:02}.png');im.thumbnail((298,424))
    card=Image.new('RGB',(320,455),'#e7eaef');card.paste(im,((320-im.width)//2,8))
    ImageDraw.Draw(card).text((12,435),f'PAGE {i+1:02}',fill='#263348');thumbs.append(card)
    blocks=p.get_text('dict')['blocks'];issues=[]
    for b in blocks:
        if b['type']!=0:continue
        for line in b['lines']:
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox']
                if x0<-1 or y0<-1 or x1>p.rect.width+1 or y1>p.rect.height+1:issues.append(span['text'])
    pages.append({'page':i+1,'text_chars':len(p.get_text()),'images':len(p.get_images()),'outside_page_text':issues})
for start in range(0,len(thumbs),6):
    canvas=Image.new('RGB',(960,910),'#dce1e8')
    for j,im in enumerate(thumbs[start:start+6]):canvas.paste(im,((j%3)*320,(j//3)*455))
    canvas.save(out/f'contact-{start//6+1:02}.png')
text='\n'.join(p.get_text() for p in doc)
(out/'extracted-text.txt').write_text(text,encoding='utf-8')
report={'pdf':str(pdf),'pages':len(doc),'pages_diagnostics':pages,'replacement_char_count':text.count('\ufffd'),
        'unresolved_reference_markers':len(re.findall(r'\[\?\]|\?\?',text))}
(out/'qa-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
