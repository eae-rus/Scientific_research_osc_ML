"""Build the active supporting figures for chapter4.

Trained KAN paths are copied from the source SVG; the training overview is
conceptual. No model or weights are loaded. Superseded architecture sketches
and numerical replots were removed in D04-010 after source-figure replacement.
"""
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from math import atan2, cos, sin, pi
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript/figures"
FONT = "ArchitectureArial"
W = 680


class Diagram:
    def __init__(self, filename, height):
        self.c = canvas.Canvas(str(OUT / filename), pagesize=(W, height))
        self.c.setTitle(filename.removesuffix(".pdf"))
        self.c.setAuthor("Dissertation architecture diagrams")
        self.h = height

    def box(self, x, y, width, text, height=48, physical=False, size=15):
        c = self.c
        c.setStrokeColor(HexColor("#263746"))
        c.setFillColor(HexColor("#E5EFEA" if physical else "#EDF1F5"))
        c.roundRect(x, y, width, height, 5, fill=1, stroke=1)
        lines = text.split("\n")
        c.setFillColor(HexColor("#182B3A"))
        c.setFont(FONT, size)
        for i, line in enumerate(lines):
            assert pdfmetrics.stringWidth(line, FONT, size) < width - 10, (line, width)
            c.drawCentredString(x + width / 2, y + height / 2 + (len(lines)-1)*size*.57 - i*size*1.14 - size/3, line)

    def arrow(self, points):
        c = self.c
        c.setStrokeColor(HexColor("#263746"))
        c.setFillColor(HexColor("#263746"))
        c.setLineWidth(1.25)
        path = c.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        c.drawPath(path)
        x, y = points[-1]
        a = atan2(y-points[-2][1], x-points[-2][0])
        p = c.beginPath(); p.moveTo(x, y)
        for sign in [-1, 1]:
            p.lineTo(x-7*cos(a+sign*pi/7), y-7*sin(a+sign*pi/7))
        p.close(); c.drawPath(p, stroke=0, fill=1)

    def text(self, x, y, value, size=14):
        self.c.setFont(FONT, size)
        self.c.setFillColor(HexColor("#263746"))
        self.c.drawCentredString(x, y, value)

    def finish(self):
        self.c.showPage(); self.c.save()


def learned_functions():
    """Keep source curve geometry; omit unreliable automatic shape names."""
    root = ET.parse(ROOT.parents[1] / "reports/Exp_2_5_and_start_Exp_2_6/figures_article1/fig11_kan_splines/fig11_kan_splines_article.svg").getroot()
    ns = {"s": "http://www.w3.org/2000/svg"}
    clips = {n.attrib['id']: next(iter(n)).attrib for n in root.findall('.//s:clipPath', ns)}
    d = Diagram("ch04_learned_kan_functions.pdf", 395)
    d.text(340, 372, "Примеры обученных функций на внутренних связях PhysicsKAN", 16)
    for pos, rank in enumerate([1, 2, 5, 9]):
        group = root.find(f'.//s:g[@id="axes_{rank}"]', ns)
        paths = [p for p in group.findall('.//s:path', ns) if p.get('clip-path')]
        curve = next(p for p in paths if 'stroke: #9467bd' in p.get('style', ''))
        clip = clips[curve.get('clip-path')[5:-1]]
        sx, sy, sw, sh = [float(clip[k]) for k in ['x','y','width','height']]
        x0, y0, w, h = 48+(pos%2)*333, 213-(pos//2)*165, 275, 118
        c=d.c;c.setStrokeColor(HexColor('#9FAAB3'));c.setLineWidth(.7);c.rect(x0,y0,w,h)
        project=lambda x,y:(x0+(x-sx)/sw*w, y0+h-(y-sy)/sh*h)
        for p in paths:
            style=p.get('style','')
            if p is not curve and 'stroke: #000000' not in style:
                continue
            commands=re.findall(r'([ML])\s*([-+\d.eE]+)\s+([-+\d.eE]+)',p.get('d',''))
            assert commands and len(re.findall(r'[ML]',p.get('d','')))==len(commands)
            path=c.beginPath()
            for cmd,x,y in commands:
                pt=project(float(x),float(y))
                (path.moveTo if cmd=='M' else path.lineTo)(*pt)
            c.setStrokeColor(HexColor('#7955A1' if p is curve else '#C0C8CE'))
            c.setLineWidth(1.7 if p is curve else .6);c.drawPath(path)
        d.text(x0+w/2,y0+h+13,f"Функция {rank} исходной иллюстрации",12)
        d.text(x0+w/2,y0-17,"Входной скрытый признак",11)
    d.text(340,17,"Кривые перенесены из векторного рисунка статьи. Масштаб каждого поля свой.",12)
    d.finish()


def two_stage_training():
    d = Diagram("ch04_two_stage_training.pdf", 302)
    d.text(340, 279, "Предварительное обучение и адаптация к целевой задаче", 16)
    for x, text in [(10,"Неразмеченные\nосциллограммы"),
                    (245,"Скрытие части\nспектральных векторов"),
                    (480,"Восстановление спектра\nОбучение общих блоков")]:
        d.box(x,190,190,text,height=60,size=14)
    d.arrow([(200,220),(245,220)]);d.arrow([(435,220),(480,220)])
    d.arrow([(670,220),(677,220),(677,141),(105,141),(105,98)])
    d.text(340,153,"Перенос коэффициентов общей части сети",13)
    for x,text in [(10,"Общая часть сети\nНовый выходной блок"),
                   (245,"Дообучение по\nразмеченным данным"),
                   (480,"Оценка по другим\nосциллограммам")]:
        d.box(x,38,190,text,height=60,size=14)
    d.arrow([(200,68),(245,68)]);d.arrow([(435,68),(480,68)])
    d.text(340,10,"Первый этап восстанавливает данные. Второй учится распознавать заданные события.",12)
    d.finish()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--active-support', action='store_true',
                        help='Generate the active support figures (also the default).')
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont(FONT, 'C:/Windows/Fonts/arial.ttf'))
    learned_functions()
    two_stage_training()


if __name__ == '__main__':
    main()
