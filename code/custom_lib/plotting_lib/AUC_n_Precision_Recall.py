from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score,precision_recall_curve,average_precision_score


class Curves_AUC_PrecionnRecall(object):
    
    def __init__(self, chexpert_load,chexpert_dataset, probs, acts, list_classes,model_name=""):
      
      self.model_name = model_name
      self.list_classes = chexpert_dataset.list_classes

      self.auc_scores = {cheXpert_valid_dataset.list_classes[i]: roc_auc_score(acts[:,i],probs[:,i]) for i in range(len(cheXpert_valid_dataset.list_classes))}
      self.chexpert_load = chexpert_load
      df_probs = pd.DataFrame(probs, columns=[cl+"probs" for cl in self.list_classes])
      df_acts = pd.DataFrame(acts, columns=[cl+"acts" for cl in self.list_classes])


      probs_n_act_df = probs_n_act_df.join(df_probs)
      probs_n_act_df = probs_n_act_df.join(df_acts)

      self.acts = probs_n_act_df.groupby(['patient','study'])[[cl+"acts" for cl in self.list_classes]].max().values
      self.probs = probs_n_act_df.groupby(['patient','study'])[[cl+"probs" for cl in self.list_classes]].mean().values

        

    def __call__(self):
      color=["r","g","b"]
      f, ax = plt.subplots(2, len(self.list_classes) , sharey=True, figsize=(20,10))


      for i in range(len(self.list_classes)):
        label_name= self.list_classes[i]

        false_positive_rate, true_positive_rate, thresholds = roc_curve(acts[:,i], probs[:,i])
        roc_auc = auc(false_positive_rate, true_positive_rate)

        rad_FPR = self.chexpert_load.radiologist_auc[label_name]["FPR"];
        rad_TPR = self.chexpert_load.radiologist_auc[label_name]["TPR"]


        precision, recall, thresholds = precision_recall_curve(acts[:,i], probs[:,i])
        average_precision = average_precision_score(acts[:, i], probs[:, i])

        sensitivity_rad = self.chexpert_load.radiologist_precision_recall[label_name]["Sensitivity"]
        precision_rad = self.chexpert_load.radiologist_precision_recall[label_name]["Precision"]

        #print("recall",recall)

        below_count_auc = self.count_points_below(rad_FPR,rad_TPR,false_positive_rate,true_positive_rate)
        below_count_P_R = self.count_points_below(sensitivity_rad,precision_rad, recall,precision)

        ax[0,i].set_title( label_name+' ROC (>'+str(below_count_auc)+')')
        ax[0,i].plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc, c="grey")

        for rad in range(3):
          ax[0,i].scatter(rad_FPR[rad], rad_TPR[rad], s=50,c=color[rad],label="Rad"+str(rad+1)+" "+str((rad_FPR[rad], rad_TPR[rad])))

        ax[0,i].scatter(rad_FPR[-1],rad_TPR[-1],s=50,c="darkblue",marker="x",label="RadMaj"+ str((rad_FPR[-1],rad_TPR[-1])))

        ax[0,i].legend(loc = 'lower right') 
        ax[0,i].axis(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.01)
        ax[0,i].set_xlabel('False Positive Rate')
        ax[0,i].set_ylabel('True Positive Rate')


        ax[1,i].set_title( label_name+' Precision_Recall(>'+str(below_count_P_R)+')')
        ax[1,i].plot( recall, precision, 'b',label='AP = %0.2f'% average_precision,c="grey")

        for rad in range(3):
          ax[1,i].scatter(sensitivity_rad[rad],precision_rad[rad],s=50,c=color[rad],label="Rad"+str(rad+1)+" "+str((sensitivity_rad[rad],precision_rad[rad])))

        ax[1,i].scatter(sensitivity_rad[-1],precision_rad[-1],s=50,c="darkblue",marker="x",label="RadMaj"+str((sensitivity_rad[-1],precision_rad[-1])))

        ax[1,i].legend()
        ax[1,i].axis(xmin=0.0,xmax=1.0,ymin=0.0,ymax=1.01)
        ax[1,i].set_ylabel('Precision')
        ax[1,i].set_xlabel('Sensitivity')
        plt.show()
        #print("recall", recall)
        #print("FPR", false_positive_rate)


    plt.savefig("saved_AUC_and_P_R.pdf")
    
    def auc_difference_print():
      #average results reported in the associated paper
      chexpert_auc_scores = {'Atelectasis':      0.858,
                             'Cardiomegaly':     0.854,
                             'Consolidation':    0.939,
                             'Edema':            0.941,
                             'Pleural Effusion': 0.936}

      max_feat_len = max(map(len, chexpert_targets))

      avg_chexpert_auc = sum(list(chexpert_auc_scores.values()))/len(chexpert_auc_scores.values())
      avg_auc          = sum(list(self.auc_scores.values()))/len(self.auc_scores.values())

      [print(f'{k: <{max_feat_len}}\t auc: {self.auc_scores[k]:.3}\t chexpert auc: {chexpert_auc_scores[k]:.3}\t difference:\
      {(chexpert_auc_scores[k]-self.auc_scores[k]):.3}') for k in chexpert_targets]

      print(f'\nAverage auc: {avg_auc:.3} \t CheXpert average auc {avg_chexpert_auc:.3}\t Difference {(avg_chexpert_auc-avg_auc):.3}')

    
#print(auc_scores)


    def count_points_below(self,x_points,y_points,x,y,num_rads=4):
      n = len(x)
      count=0
      #3 rad and RadMaj
      for rad_i in range(num_rads):

        y_point=y_points[rad_i]
        x_point=x_points[rad_i]

        eq = np.where(x==x_point)[0]
        if eq.size:
          idx = eq[0]
          y_closest = y[idx]

        else:

          if np.all((np.diff(x)>=0)):#if ordered in ascending order
            idx_nextbig = np.argmax(x > x_point) #since they are ordered its gonna give closest numbers possible
            idx_nextsmall = n-np.argmax(np.flip(x) < x_point)-1

          else:#if ordered in descending order
            idx_nextsmall = np.argmax(x < x_point)
            idx_nextbig = n-np.argmax(np.flip(x) > x_point)-1

          y_big = y[idx_nextbig]
          y_small =y[idx_nextsmall]

          x_big = x[idx_nextbig]
          x_small =x[idx_nextsmall]

          y_closest = y_small+ (y_big-y_small) * ((x_point-x_small)/(x_big-x_small))

        if y_point < y_closest :  
          #above the line
           count = count + 1

      return count

     

#how to check points above or below line