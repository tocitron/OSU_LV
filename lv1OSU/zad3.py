brojevi = []

while True:
    unos=input("Unesite broj(ili 'Done za kraj'): ")
    
    if unos == "Done":
        break

    try:
        broj=float(unos)
        brojevi.append(broj)
    except:
        print("Pogrešan unos! Molimo unesite broj.")

if len(brojevi) == 0:
    print("Niste unijeli nijedan broj.")
else:
    print("Broj unesenih brojeva:", len(brojevi))
    print("Srednja vrijednost:", sum(brojevi)/len(brojevi))
    print("Min vrijednost:", min(brojevi))
    print("Max vrijednost:", max(brojevi))

    brojevi.sort()
    print("Sortirana lista:", brojevi)