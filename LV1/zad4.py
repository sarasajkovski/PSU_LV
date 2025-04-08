def main():
    try:
        ime_datoteke = input("Ime datoteke: ").strip()
        with open(ime_datoteke, 'r') as datoteka:
            ukupno = 0.0
            brLinija = 0
            
            for linija in datoteka:
                if linija.startswith("X-DSPAM-Confidence:"):
                    dio = linija.split(":")[1].strip()
                    try:
                        broj = float(dio)
                        ukupno += broj
                        brLinija += 1
                    except ValueError:
                        continue

            if brLinija > 0:
                prosjek = ukupno / brLinija
                print("Average X-DSPAM-Confidence:", prosjek)
            else:
                print("Ne postoji X-DSPAM-Confidence linija.")

    except FileNotFoundError:
        print("Datoteka nije pronađena.")
        
if __name__ == "__main__":
    main()