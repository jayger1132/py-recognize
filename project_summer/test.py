import pymysql
Grade = 0
tempA = [1,1,2,3,4,4,4,4,4,4,3,2,2,1,1 ]
tempB = [1,1,2,3,4,4,4,4,4,4,3,2,2,1,1 ]
for i in range(0,len(tempA)) :
    if i<=len(tempA)/4:
        if(tempA[i]==1 or tempA[i]==2):
            Grade+=tempA[i] + tempB[i]
    elif i<=3*len(tempA)/4:
        if(tempA[i]==3 or tempA[i]==4):
            Grade+=tempA[i] + tempB[i]
    else:
        if(tempA[i]==1 or tempA[i]==2):
            Grade+=tempA[i] + tempB[i]
    print(Grade)
output = str(Grade/len(tempA)*100) +"%"
print("評分為",Grade/len(tempA)*100,"%")
temp = int(Grade/len(tempA)*100)