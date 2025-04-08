import string
def main():
    try:
        with open("song.txt", "r") as file:
            tekst = file.read()

        tekst = tekst.lower()
        for znak in string.punctuation:
            tekst = tekst.replace(znak, "")

        rijeci = tekst.split()
        rjecnik = {}
        for rijec in rijeci:
            rjecnik[rijec] = rjecnik.get(rijec, 0) + 1

        jednom_rijeci = [rijec for rijec, broj in rjecnik.items() if broj == 1]
        print(f"Broj riječi koje se pojavljuju samo jednom: {len(jednom_rijeci)}")
        print("Riječi:")
        for rijec in jednom_rijeci:
            print(rijec)

    except FileNotFoundError:
        print("Datoteka nije pronađena.")

if __name__ == "__main__":
    main()
