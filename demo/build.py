# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow", "fpdf2"]
# ///
"""Builds the folder the README's demo searches: a home with photos, a screenshot, a lease PDF, notes, a podcast and
a video, all made here or public domain (demo/photos, see THIRD_PARTY_NOTICES.md).

    uv run demo/build.py <home>      # e.g. ~/.cache/rummage-demo/home (outside any code project)

Speech comes from `say` (macOS) or `espeak-ng`; audio and video need ffmpeg.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont

home = pathlib.Path(sys.argv[1]).expanduser().resolve()
here = pathlib.Path(__file__).resolve().parent
if home.exists():
    shutil.rmtree(home)
DAY = 86400
now = time.time()


def place(path: pathlib.Path, days_ago: float):
    os.utime(path, (now - days_ago * DAY, now - days_ago * DAY))


def write(relative: str, text: str, days_ago: float):
    path = home / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    place(path, days_ago)


# Photos, named as a phone names them: found by what's in them, not by name.
pictures = home / "Pictures/2025"
pictures.mkdir(parents=True)
for i, photo in enumerate(sorted((here / "photos").glob("IMG_*.jpg"))):
    shutil.copy(photo, pictures / photo.name)
    place(pictures / photo.name, 40 + i * 9)


# A screenshot of a checkout that failed.
def font(size, bold=False):
    for path in (["/System/Library/Fonts/Supplemental/Arial Bold.ttf"] if bold else []) + [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


shot = Image.new("RGB", (1440, 900), "#f5f5f7")
draw = ImageDraw.Draw(shot)
draw.rectangle([0, 0, 1440, 52], fill="#e6e6ea")
for x, colour in ((22, "#ff5f57"), (44, "#febc2e"), (66, "#28c840")):
    draw.ellipse([x - 6, 20, x + 6, 32], fill=colour)
draw.text((620, 16), "checkout.northwind-supplies.com", font=font(18), fill="#555")
draw.text((120, 110), "Northwind Supplies", font=font(34, True), fill="#1d1d1f")
draw.text((120, 160), "Checkout · Order #NW-48213", font=font(22), fill="#6e6e73")
draw.rectangle([120, 220, 1320, 330], fill="#fde8e8", outline="#f5b5b5", width=2)
draw.text((150, 240), "Payment failed: your card was declined", font=font(30, True), fill="#b42318")
draw.text((150, 286), "The bank declined the charge (card_declined). Try another card or contact your bank.",
          font=font(21), fill="#7a271a")
rows = [("Toner cartridge, black × 2", "€183.00"), ("Recycled A4 paper, 5 reams × 4", "€152.00"),
        ("Delivery", "€9.50"), ("Total", "€344.50")]
for k, (item, price) in enumerate(rows):
    y = 380 + k * 56
    draw.text((150, y), item, font=font(24, item == "Total"), fill="#1d1d1f")
    draw.text((1150, y), price, font=font(24, item == "Total"), fill="#1d1d1f")
    draw.line([150, y + 44, 1290, y + 44], fill="#d2d2d7")
draw.rounded_rectangle([150, 640, 470, 704], radius=10, fill="#0071e3")
draw.text((198, 657), "Try another card", font=font(24, True), fill="white")
screenshots = home / "Pictures/Screenshots"
screenshots.mkdir(parents=True)
shot_path = screenshots / "Screenshot 2026-09-14 at 10.41.07.png"
shot.save(shot_path)
place(shot_path, 35)

# A lease, the late fee on page 4.
clauses = [
    ("1. Parties and property", "This agreement is between Harbour Lettings Ltd (the landlord) and the tenant named "
     "below, for the flat at 14 Harbour Road, second floor, including the fitted kitchen, the bathroom fixtures and "
     "the storage cupboard on the landing."),
    ("2. Term", "The tenancy runs for twelve months from 1 March 2026. Either party may end it after six months with "
     "two months' notice in writing. If neither does, it continues from month to month on the same terms."),
    ("3. Rent", "The rent is £1,450 a month, paid in advance on the first day of each month by bank transfer to the "
     "account the landlord gives in writing. The rent may be reviewed once a year with two months' notice."),
    ("4. Late payment", "If the rent is more than five days late, the tenant pays a late fee of £45, plus £5 for each "
     "further day it remains unpaid, up to £150 a month. Payments are applied to late fees first, then to rent."),
    ("5. Deposit", "The tenant pays a deposit of £2,175, held in a government-approved protection scheme. It is "
     "returned within ten working days of the final inspection, less the cost of any damage beyond fair wear."),
    ("6. Pets", "No pets may be kept without the landlord's written consent, which will not be unreasonably "
     "withheld for a cat or a small dog. Consent may require a professional clean when the tenancy ends."),
    ("7. Repairs", "The landlord keeps the structure, the heating, the hot water and the wiring in repair. The tenant "
     "reports problems promptly and allows access for repairs at reasonable times with 24 hours' notice."),
    ("8. Ending the tenancy", "At the end, the tenant returns all keys, leaves the flat clean, and gives a forwarding "
     "address. Meter readings are taken on the last day, and the final inspection follows within three days."),
]
pdf = FPDF()
pdf.set_margins(25, 25)
for page, (title, text) in enumerate(clauses):
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Residential Tenancy Agreement", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"14 Harbour Road · page {page + 1} of {len(clauses)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 7, text)
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, "Initialled by the landlord and the tenant. This page forms part of the agreement and "
                   "should be read with the schedule of fixtures and the inventory signed on the move-in date.")
lease = home / "Documents/Lease - 14 Harbour Road.pdf"
lease.parent.mkdir(parents=True)
pdf.output(str(lease))
place(lease, 220)

# Notes.
write("Documents/Notes/sourdough.md", """# Sourdough, the way that finally worked

Feed the starter the night before. Mix 500 g flour with 350 g water and leave it for an hour, then add 100 g of
starter and 10 g of salt. Fold every half hour for two hours, then leave it alone until it has grown by half, around
five hours in a warm kitchen. Shape, rest in the fridge overnight, bake at 250 °C in a covered pot for 20 minutes,
then uncovered for 25.
""", 12)
write("Documents/Notes/garden plan.md", """# Garden plan for spring

Raised bed one: tomatoes along the back fence where they get afternoon sun, basil between them.
Raised bed two: carrots and radishes, sown in rows a hand apart. Net them against the pigeons.
Order seed potatoes in February and chit them on the windowsill for six weeks before planting.
""", 70)
write("Documents/Notes/Lisbon trip.md", """# Lisbon, May

Flights booked: out Friday 8 May, back Tuesday 12 May. Staying in Alfama, near the castle.
Must do: tram 28 early before the crowds, pastéis de nata in Belém, the tile museum, sunset at Miradouro da Senhora
do Monte.
""", 25)
write("Documents/Notes/standup.md", """# Standup notes

Shipped the export button. Next: fix the flaky login test. Blocked on design for the empty state.
""", 2)
write("Documents/Notes/books.md", """# Books to read

- The Overstory, Richard Powers
- Piranesi, Susanna Clarke
- The Dispossessed, Ursula K. Le Guin
- Braiding Sweetgrass, Robin Wall Kimmerer
""", 90)


# Speech: `say` on macOS, espeak-ng elsewhere.
def speak(text: str, out: pathlib.Path):
    aiff = out.with_suffix(".aiff")
    if shutil.which("say"):
        subprocess.run(["say", "-v", "Samantha", "-r", "175", "-o", str(aiff), text], check=True)
    elif shutil.which("espeak-ng"):
        subprocess.run(["espeak-ng", "-s", "160", "-w", str(aiff), text], check=True)
    else:
        sys.exit("needs `say` (macOS) or espeak-ng for the podcast")
    return aiff


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


work = pathlib.Path(tempfile.mkdtemp())

# A podcast: the part about caffeine starts about a minute and a half in.
segments = [
    "Welcome back to Sleep Better, the show about getting a good night's rest. I'm your host. This week we're talking "
    "about the small things in the evening that decide how well you sleep: the temperature of your bedroom, what you "
    "drink in the afternoon, and the light from your screens. Let's start with the bedroom.",
    "Most people sleep best when the bedroom is cool, somewhere around seventeen or eighteen degrees. Your body "
    "temperature drops as you fall asleep, and a warm room fights that. A lighter duvet, an open window, or a fan can "
    "make more difference than any supplement. Socks, oddly enough, help too: warm feet let the body shed heat.",
    "Now, coffee. Caffeine has a half-life of about five to six hours, so half of the coffee you drink at three in "
    "the afternoon is still in your body at nine at night. It blocks adenosine, the chemical that builds sleep "
    "pressure through the day. If you struggle to fall asleep, try a cut-off at noon for two weeks and see what changes. "
    "Tea and dark chocolate count too.",
    "Finally, screens. It isn't only the blue light: it's that a phone keeps your mind busy. Put it on the other side "
    "of the room, read a paper book, and keep the same wake-up time every day, weekends included. That's it for this "
    "week. Sleep well.",
]
parts = []
for k, text in enumerate(segments):
    parts.append(speak(text, work / f"part{k}.aiff"))
listing = work / "parts.txt"
listing.write_text("".join(f"file '{p}'\nfile '{here / 'silence.wav'}'\n" for p in parts))
ffmpeg("-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono", "-t", "1.5", str(here / "silence.wav"))
podcast = home / "Podcasts/Sleep Better - episode 12.m4a"
podcast.parent.mkdir(parents=True)
ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-ac", "1", "-ar", "22050", "-c:a", "aac", "-b:a", "64k",
       str(podcast))
(here / "silence.wav").unlink()
place(podcast, 18)

# A video: a title card, then the cabin in the snow, with a voice over.
card = Image.new("RGB", (1280, 720), "#1c2733")
ImageDraw.Draw(card).text((640, 360), "Winter weekend", font=font(72, True), fill="white", anchor="mm")
card.save(work / "card.png")
cabin = Image.open(here / "photos/cabin.jpg").convert("RGB")
cabin = cabin.resize((1280, int(cabin.height * 1280 / cabin.width))).crop((0, 0, 1280, 720))
cabin.save(work / "cabin.png")
voice = speak("We drove up on Friday night. The cabin was buried in snow, and it took an hour to dig out the "
              "door. Worth it.", work / "voice.aiff")
video = home / "Movies/Winter weekend.mp4"
video.parent.mkdir(parents=True)
ffmpeg("-loop", "1", "-t", "6", "-i", str(work / "card.png"), "-loop", "1", "-t", "12", "-i", str(work / "cabin.png"),
       "-i", str(voice),
       "-filter_complex",
       "[1:v]zoompan=z='1+0.0015*on':d=1:s=1280x720:fps=25[z];[0:v]fps=25,format=yuv420p[a];"
       "[z]format=yuv420p[b];[a][b]concat=n=2:v=1:a=0[v];[2:a]adelay=6000|6000,apad[s]",
       "-map", "[v]", "-map", "[s]", "-c:v", "libx264", "-crf", "26", "-c:a", "aac", "-t", "18", str(video))
place(video, 60)

shutil.rmtree(work)
print(f"built {home}")
