#sati=float(input("Radni sati: "))
#satnica=float(input("eura/h: "))
#zarada=sati*satnica
#print("Ukupno:", zarada, "eura")



def total_euro(sati, satnica):
    return sati*satnica
sati=float(input("Radni sati: "))
satnica=float(input("eura/h: "))

ukupno=total_euro(sati, satnica)
print("Ukupno:", ukupno, "eura")