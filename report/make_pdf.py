"""Render the Phase 1 report to PDF with reportlab (no LaTeX needed).

Two-column academic layout, embedded figures, matches report.tex content.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image,
    Table, TableStyle, FrameBreak, NextPageTemplate, KeepTogether,
)
from reportlab.platypus.flowables import HRFlowable
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
OUT = os.path.join(HERE, "Phase1_Report.pdf")

# ---- styles ---------------------------------------------------------------
ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Times-Roman",
                      fontSize=9, leading=11.5, alignment=TA_JUSTIFY,
                      spaceAfter=5)
h = ParagraphStyle("h", parent=ss["Normal"], fontName="Times-Bold",
                   fontSize=10, leading=12, spaceBefore=7, spaceAfter=3)
abhead = ParagraphStyle("abhead", parent=body, fontName="Times-BoldItalic",
                        fontSize=9)
cap = ParagraphStyle("cap", parent=body, fontSize=8, leading=9.5,
                     spaceBefore=2, spaceAfter=8)
title = ParagraphStyle("title", parent=ss["Title"], fontName="Times-Bold",
                       fontSize=16, leading=19, alignment=TA_CENTER)
authors = ParagraphStyle("auth", parent=body, fontSize=10, leading=13,
                         alignment=TA_CENTER, spaceBefore=6)

# ---- page geometry: title spans both cols, then two columns ---------------
PW, PH = letter
M = 0.6 * inch
GUT = 0.3 * inch
colw = (PW - 2 * M - GUT) / 2
top = PH - M
banner_h = 1.5 * inch  # space for title block on page 1

# frames
full_frame = Frame(M, M, PW - 2 * M, PH - 2 * M - banner_h - 0.1 * inch,
                   id="full", leftPadding=0, rightPadding=0,
                   topPadding=0, bottomPadding=0)
# page 1: banner handled by onPage, two columns below
l1 = Frame(M, M, colw, PH - 2 * M - banner_h, id="l1",
           leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
r1 = Frame(M + colw + GUT, M, colw, PH - 2 * M - banner_h, id="r1",
           leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
# later pages: full-height two columns
l2 = Frame(M, M, colw, PH - 2 * M, id="l2",
           leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
r2 = Frame(M + colw + GUT, M, colw, PH - 2 * M, id="r2",
           leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

story = []


def para(t, style=body):
    story.append(Paragraph(t, style))


def head(t):
    story.append(Paragraph(t, h))


def figure(fname, width_frac, caption):
    path = os.path.join(FIG, fname)
    from PIL import Image as PILImage
    iw, ih = PILImage.open(path).size
    w = colw * width_frac
    img = Image(path, width=w, height=w * ih / iw)
    img.hAlign = "CENTER"
    story.append(KeepTogether([Spacer(1, 3), img,
                               Paragraph(caption, cap)]))


# ============================ CONTENT =====================================
para('<b><i>Abstract</i></b>&mdash; We estimate where a phone is '
     '&mdash; which building, which floor, and its coordinates &mdash; from '
     'the strengths of the Wi-Fi access points it can hear. Working on the '
     'UJIIndoorLoc dataset, we build a deliberately plain baseline: separate '
     'tree models for each target, trained on the dataset’s own '
     'train/test split. Random Forest reads the building almost perfectly '
     '(99.9%) and gives the tightest coordinates (6.96&nbsp;m median error); '
     'XGBoost is the better floor classifier (92.0%). Routing each task to '
     'whichever model wins it yields a combined system at 12.0&nbsp;m mean / '
     '7.1&nbsp;m median positioning error. Two things stand out from the error '
     'analysis and both carry into Phase&nbsp;2: a wrong floor roughly doubles '
     'the coordinate error, and accuracy varies by more than a factor of three '
     'across unseen handsets. These numbers set the bar that a shared '
     'multitask network will have to beat.')
para('<b>Index Terms</b>&mdash; indoor localization, Wi-Fi fingerprinting, '
     'UJIIndoorLoc, Random Forest, XGBoost, multitask learning.')

head("I.&nbsp;&nbsp;INTRODUCTION")
para('Satellite positioning stops working once you walk indoors, so phones '
     'fall back on whatever signals are around &mdash; and Wi-Fi is '
     'everywhere. The idea behind fingerprinting is simple: record, at many '
     'known spots, how loud each nearby access point sounds, then match a new '
     'reading against that map. UJIIndoorLoc is the standard benchmark for '
     'this. It covers three university buildings with several floors each, and '
     'every scan lists the received signal strength (RSSI) of up to 520 access '
     'points.')
para('What makes the problem interesting is that it is really three problems '
     'stacked together. You want the building (a 3-way choice), the floor (5 '
     'possibilities), and a pair of coordinates in metres. These are not '
     'independent &mdash; knowing the building narrows the floors, knowing the '
     'floor constrains the coordinates. This report is Phase&nbsp;1: a '
     'classical machine-learning baseline that treats each of the three as its '
     'own model and measures how far that gets us. It is meant to be honest '
     'and reproducible rather than clever. Phase&nbsp;2 will replace it with a '
     'network that learns the three targets jointly; the point of a baseline '
     'is to know what “jointly” has to improve on.')

head("II.&nbsp;&nbsp;DATA AND EXPLORATION")
para('The training file holds 19,937 scans and the separate validation file '
     'holds 1,111. We use that split as given. Because the validation scans '
     'were gathered later and on different phones, it is a genuine test of '
     'generalisation rather than a reshuffle of the same recordings, and we '
     'never fit any transform on it.')
para('A first cleaning pass found 637 training scans that were exact copies '
     'of another row &mdash; same signals, same labels &mdash; so we dropped '
     'them and kept 19,300 unique scans. The first thing exploration makes '
     'obvious is how little each scan actually sees: the median fingerprint '
     'picks up only 17 of the 520 access points (Fig.&nbsp;1), and 96.5% of '
     'the whole signal matrix is empty. Fifty-five access points are never '
     'heard anywhere in training, so they carry no information and we drop '
     'their columns.')
figure("eda_aps_per_sample.png", 1.0,
       "<b>Fig. 1.</b> How many access points a single scan detects. Most see "
       "fewer than twenty of the 520 available &mdash; the data is sparse by "
       "nature.")
para('The three buildings sit in clearly separate places on the map '
     '(Fig.&nbsp;2), which is a useful hint early on: telling the buildings '
     'apart should be easy, and the real difficulty will be floors and fine '
     'positioning. The floors are not evenly represented. Floor&nbsp;4 exists '
     'only in one building and shows up in roughly a thousand scans, far fewer '
     'than the others, so we watch it carefully during evaluation.')
figure("eda_spatial_layout.png", 1.0,
       "<b>Fig. 2.</b> Training fingerprints at their true coordinates, "
       "coloured by building. The three buildings occupy distinct regions.")

head("III.&nbsp;&nbsp;METHOD")
para('<b>Preparing the signals.</b> UJIIndoorLoc marks an access point that '
     'was not heard with the value +100, which is not a real signal level and '
     'would wreck any distance or scaling if left alone. We remap it to '
     '−105&nbsp;dBm, just below the weakest genuine reading '
     '(−104), so “not heard” becomes a consistent floor value. '
     'The observed RSSI is then standardised using statistics from the '
     'training data only, and the same transform is applied unchanged to the '
     'validation set &mdash; so the validation data is free to drift rather '
     'than being forced to look normalised.')
para('<b>Features.</b> On top of the 465 surviving RSSI columns we add a '
     'matching set of 465 binary “absence” flags, one per access '
     'point, marking whether it was heard. Once the signals are standardised, '
     'a flag is the only thing left that still separates “weak but '
     'heard” from “silent,” and that distinction turns out to '
     'matter. We also compute a handful of per-scan summaries &mdash; how many '
     'access points were heard, the strongest reading and which access point '
     'produced it, and the mean, spread, and range of the heard signals. '
     'Altogether the design matrix has 936 columns. Everything here is cheap '
     'to compute and easy to explain, which is the whole idea at this stage.')
para('<b>Models.</b> For each of the three targets we train both a Random '
     'Forest and an XGBoost model: classifiers for building and floor, '
     'regressors for the two coordinates (predicted together). That is six '
     'estimators in total, each ignorant of the others &mdash; which is '
     'exactly the gap Phase&nbsp;2 is meant to close. Hyperparameters are left '
     'at sensible fixed values; we are not tuning a competition entry, we are '
     'drawing a baseline.')

head("IV.&nbsp;&nbsp;RESULTS")
para('<b>Classification and position.</b> Table&nbsp;I gives the full '
     'picture. Building recognition is effectively solved &mdash; Random '
     'Forest misses 1 scan in 1,111. Floors are harder, and the two models '
     'split the honours: XGBoost reaches 92.0% floor accuracy against Random '
     'Forest’s 90.7%, while Random Forest gives noticeably tighter '
     'coordinates (6.96&nbsp;m median versus 10.06&nbsp;m). Neither model wins '
     'everything, so the natural thing is to route each task to its stronger '
     'model &mdash; building and coordinates from Random Forest, floor from '
     'XGBoost. That combined system (last row of Table&nbsp;I) gives the best '
     'figure on every column without training anything new: 12.0&nbsp;m mean '
     'and 7.1&nbsp;m median error, with a 90th percentile just under 25&nbsp;m.')

# Table I
tbl_data = [
    ["Model", "Build.", "Floor", "Coord.", "Coord."],
    ["", "acc.", "acc.", "mean (m)", "med. (m)"],
    ["Random Forest", "0.999", "0.907", "11.66", "6.96"],
    ["XGBoost", "0.997", "0.920", "14.52", "10.06"],
    ["Combined", "0.999", "0.920", "11.66", "6.96"],
]
tbl = Table(tbl_data, colWidths=[colw * x for x in (0.34, 0.15, 0.15, 0.18, 0.18)])
tbl.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 1), "Times-Bold"),
    ("FONTNAME", (0, 2), (-1, -1), "Times-Roman"),
    ("FONTNAME", (0, 4), (-1, 4), "Times-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("ALIGN", (0, 0), (0, -1), "LEFT"),
    ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
    ("LINEBELOW", (0, 1), (-1, 1), 0.5, colors.black),
    ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
    ("TOPPADDING", (0, 0), (-1, -1), 1.5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
]))
story.append(KeepTogether([
    Spacer(1, 3),
    Paragraph("<b>TABLE I.</b>&nbsp;&nbsp;VALIDATION RESULTS. "
              "“COMBINED” ROUTES EACH TASK TO ITS BEST MODEL.",
              ParagraphStyle("tc", parent=cap, alignment=TA_CENTER)),
    tbl, Spacer(1, 6)]))

para('The floor confusion matrix (Fig.&nbsp;3) explains where the remaining '
     'floor errors come from: they are almost all off-by-one, a scan placed on '
     'the floor directly above or below its true one. That matches physical '
     'intuition &mdash; a signal bleeds through a ceiling far more easily than '
     'through three of them. The rare Floor&nbsp;4 is the weakest class, '
     'recalled 79.5% of the time by XGBoost and only 71.8% by Random Forest. '
     'We tried rebalancing the class weights to help it and it made things '
     'slightly worse, which confirmed the problem is confusion with the '
     'adjacent floor, not simple under-prediction, so we left the plain model '
     'in place.')
figure("confusion_floor_XGBoost.png", 1.0,
       "<b>Fig. 3.</b> Floor confusion for XGBoost. Mistakes cluster on the "
       "diagonal neighbours &mdash; almost every error is a single floor up or "
       "down.")
para('<b>Where the error lives.</b> Plotting the coordinate error at each true '
     'location (Fig.&nbsp;4) shows it is not spread evenly. Corridors with '
     'dense fingerprint coverage are accurate; the large errors sit at '
     'building edges and corners where fewer access points reach. The error '
     'CDF (Fig.&nbsp;5) makes the model comparison clean &mdash; Random '
     'Forest’s curve sits to the left of XGBoost’s at every '
     'percentile, so it is not just better on average but better across the '
     'whole distribution of coordinate errors.')
figure("error_map_RandomForest.png", 1.0,
       "<b>Fig. 4.</b> Coordinate error at each validation point. Error "
       "concentrates at building perimeters where coverage thins out.")
figure("error_cdf_comparison.png", 1.0,
       "<b>Fig. 5.</b> Cumulative distribution of coordinate error. Random "
       "Forest dominates at every distance.")
para('One number ties the three tasks together. When the floor is predicted '
     'correctly, the mean coordinate error is 10.2&nbsp;m; when it is wrong, '
     'it jumps to 25.4&nbsp;m. The targets are coupled &mdash; a '
     'classification slip drags the regression down with it &mdash; and a '
     'model that predicted all three at once could in principle stop that from '
     'happening. That is the single clearest argument for Phase&nbsp;2.')
para('<b>Robustness across devices.</b> Averages hide a lot. The validation '
     'set uses eleven phone models, none of them seen in training, and the '
     'per-phone breakdown (Fig.&nbsp;6) shows the error swinging from under '
     '8&nbsp;m on the best handset to about 31&nbsp;m on the worst &mdash; a '
     'spread of roughly four to one for Random Forest. One phone in particular '
     'is poor for both models. XGBoost, while less accurate on average, is a '
     'little more even across devices. For a system that would actually ship '
     'on many phones, this device sensitivity is as important as the headline '
     'accuracy, and it is not something a single mean error will ever reveal.')
figure("robustness_by_phone.png", 1.0,
       "<b>Fig. 6.</b> Mean coordinate error per phone model, none seen during "
       "training. Accuracy varies more than three-fold across handsets.")

head("V.&nbsp;&nbsp;DISCUSSION AND NEXT STEPS")
para('The baseline does its job: building is essentially solved, floors land '
     'in the low nineties, and median positioning sits around 7&nbsp;m, all '
     'from models that are easy to reason about and quick to retrain. The '
     'honest limits are also clear. We report on a single supplied split '
     'rather than repeated folds, so the figures are a point estimate, not a '
     'confidence interval. We did no hyperparameter search. And the three '
     'models never talk to each other.')
para('That last point is the hand-off to Phase&nbsp;2. The coupling between '
     'floor and coordinate error, and the uneven behaviour across devices, '
     'both suggest that a shared representation &mdash; one trunk feeding a '
     'building head, a floor head and a coordinate head &mdash; has room to '
     'help where separate models cannot. We will keep the same preprocessing, '
     'the same features, and the same split and metrics, so whatever '
     'Phase&nbsp;2 reports can be compared against this table directly and '
     'fairly.')

head("REFERENCES")
para('[1] J. Torres-Sospedra et al., “UJIIndoorLoc: A new multi-building '
     'and multi-floor database for WLAN fingerprint-based indoor localization '
     'problems,” in Proc. IPIN, 2014.', ParagraphStyle(
         "ref", parent=body, fontSize=8, leading=10))
para('[2] R. Caruana, “Multitask learning,” Machine Learning, '
     'vol. 28, no. 1, pp. 41–75, 1997.', ParagraphStyle(
         "ref", parent=body, fontSize=8, leading=10))


# ---- page-1 banner painter -----------------------------------------------
def draw_banner(canvas, doc):
    canvas.saveState()
    y = PH - M - 6
    canvas.setFont("Times-Bold", 15)
    for line in ["Predicting Indoor Location from Wi-Fi Signal Strength:",
                 "A Phase 1 Baseline on UJIIndoorLoc"]:
        canvas.drawCentredString(PW / 2, y, line)
        y -= 19
    canvas.setFont("Times-Roman", 10)
    y -= 6
    canvas.drawCentredString(PW / 2, y,
                             "Aviral Mishra, Ved Pawar, Khuswant Rajpurohit, "
                             "Prataya Silla, Vansh Jain")
    y -= 13
    canvas.setFont("Times-Italic", 9)
    canvas.drawCentredString(PW / 2, y, "Project 9 — Applied Machine Learning")
    y -= 6
    canvas.setLineWidth(0.5)
    canvas.line(M, y, PW - M, y)
    canvas.restoreState()


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8)
    canvas.drawCentredString(PW / 2, M - 18, str(doc.page))
    canvas.restoreState()


def page1(canvas, doc):
    draw_banner(canvas, doc)
    footer(canvas, doc)


doc = BaseDocTemplate(OUT, pagesize=letter)
doc.addPageTemplates([
    PageTemplate(id="first", frames=[l1, r1], onPage=page1),
    PageTemplate(id="rest", frames=[l2, r2], onPage=footer),
])
# after first page, switch to rest
story.insert(0, NextPageTemplate("rest"))
doc.build(story)
print(f"wrote {OUT}")
