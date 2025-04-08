def total_euro(sati, iznos):
    return sati * iznos

sati = float(input("Radni sati: "))
iznos = float(input("Eura/h: "))
ukupno = total_euro(sati, iznos)

print(f"Ukupno: {ukupno} eura")

