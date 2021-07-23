import os
path='./data/imgs/from_video/Biceps_curl\\straight' #這就是欲進行檔名更改的檔案路徑，路徑的斜線是為/，要留意下！
files=os.listdir(path)
#print(files) #印出讀取到的檔名稱，用來確認自己是不是真的有讀到

#設定初始值
p = 0
for i in files :
    oldname=path+"/"+i
    print(oldname)
    newname = path + '/' + str(p) + ".jpg"
    print(newname)
    p = p+1
    os.renames(oldname, newname)
##更改最外圍資料夾檔名
#for i in files: #因為資料夾裡面的檔案都要重新更換名稱
#    oldname=path+"/"+files[n] #指出檔案現在的路徑名稱，[n]表示第n個檔案
#    print (path,'\t',files[n])
    
    
#    if(files[n]):
#        newname = path + '/test_' + str(o)
#        os.rename(oldname,newname)
#        o = o+1
#        print ('newname \t',newname)
#    #更改裡面圖片

#    path1 = newname
#    file1 = os.listdir(path1)
#    print (path1,"\t",file1)
#    for i in file1 :
#        path2 = path1 +"\\"+i
#        file2 = os.listdir(path2)
#        print("lower \t" ,file2)
#        inner=10 #更改內層檔案的檔案名稱
#        p=0
#        for i in file2:
#            oldname = path2+'/'+file2[p]
#            print (oldname)
#            newname = path2+'/'+str(inner)+".bmp"
#            print(newname)
#            os.renames(oldname, newname)
#            inner+=1
#            p+=1
#    n=n+1


'''
    newname=path+'2017-'+str(n+1)+'.wav' #在本案例中的命名規則為：年份+ - + 次序，最後一個.wav表示該檔案的型別
	os.rename(oldname,newname)
	print(oldname+'>>>'+newname) #印出原名與更名後的新名，可以進一步的確認每個檔案的新舊對應
	n=n+1 #當有不止一個檔案的時候，依次對每一個檔案進行上面的流程，直到更換完畢就會結束
'''