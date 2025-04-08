def main():
    brojevi = []
    while True:
        unos = input("Unesite broj ('Done' za kraj): ")
        if unos.lower() == "done":
            break

        broj = float(unos)
        brojevi.append(broj)

    if brojevi:
        print(f"Broj unesenih brojeva: {len(brojevi)}" )
        print(f"Srednja vrijednost: {sum(brojevi) / len(brojevi): }")
        print(f"Minimalna vrijednost: {min(brojevi)}")
        print(f"Maksimalna vrijednost: {max(brojevi)}")
        brojevi.sort()
        print("Sortirana lista:", brojevi)
    else:
        print("Greška.")


if __name__ == "__main__":
    main()