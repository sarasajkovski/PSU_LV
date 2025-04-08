def main():
    try:
        with open("SMSSpamCollection.txt", "r", encoding="utf-8") as file:
            ham_broj_poruka = 0
            ham_ukupno_rijeci = 0
            spam_broj_poruka = 0
            spam_ukupno_rijeci = 0
            spam_zavrsava_usklicnik = 0

            for linija in file:
                linija = linija.strip()
                if not linija:
                    continue  

                if '\t' in linija:
                    tip, poruka = linija.split('\t', 1)
                else:
                    continue 

                broj_rijeci = len(poruka.split())

                if tip.lower() == "ham":
                    ham_broj_poruka += 1
                    ham_ukupno_rijeci += broj_rijeci
                elif tip.lower() == "spam":
                    spam_broj_poruka += 1
                    spam_ukupno_rijeci += broj_rijeci
                    if poruka.strip().endswith("!"):
                        spam_zavrsava_usklicnik += 1

            prosjek_ham = ham_ukupno_rijeci / ham_broj_poruka if ham_broj_poruka else 0
            prosjek_spam = spam_ukupno_rijeci / spam_broj_poruka if spam_broj_poruka else 0
            print(f"(a) Prosječan broj riječi u ham porukama: {prosjek_ham:.2f}")
            print(f"(a) Prosječan broj riječi u spam porukama: {prosjek_spam:.2f}")
            print(f"(b) Broj spam poruka koje završavaju uskličnikom: {spam_zavrsava_usklicnik}")

    except FileNotFoundError:
        print("Datoteka  nije pronađena.")

if __name__ == "__main__":
    main()