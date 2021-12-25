import os
action = "Side_Lateral_RaiseR"
recognize_path = "./img/"+action+"/"
try:
  os.makedirs(recognize_path)
# 檔案已存在的例外處理
except FileExistsError:
  print("檔案已存在。")