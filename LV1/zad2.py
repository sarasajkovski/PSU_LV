def main():
    try:
        ocjena = float(input("Unesite ocjenu ( između 0.0 i 1.0): "))

        if 0.0 <= ocjena <= 1.0:
            if ocjena >= 0.9:
                print("A")
            elif ocjena >= 0.8:   
                print("B")
            elif ocjena >= 0.7:
                print("C") 
            elif ocjena >= 0.6:
                print("D")
            elif ocjena < 0.6:
                print("F")         
        else:
            print("Ocjena mora biti između 0.0 i 1.0.")    

    except ValueError:
        print("Greška") 

if __name__ == "__main__":
    main()