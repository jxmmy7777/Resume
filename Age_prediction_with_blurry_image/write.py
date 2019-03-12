import csv
all_value=[[i for i in range(1:config.epoch+1)],train_loss_list,train_acc_list,val_loss_list,val_acc_list]
title=str(epoch)+","+str(train_loss_list)+","+str(train_acc_list)+","+str(val_loss_list)+","+str(val_acc_list)
row_value=zip(*all_value)
sav_value+=title+"\n"
for i in range(len(row_value)):
    data+=str(row_value[i][0])+","+str(row_value[i][1])+"\n"

with open("data.csv", "w") as f1:
    f1.write(sav_value)


