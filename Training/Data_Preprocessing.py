# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:15:31 2017

@author: laramos
"""

import pandas as pd
import numpy as np
#from fancyimpute import MICE

"""
These methods below only select the variables per study.

"""

def Get_Vars_Baseline_binscores(frame):  
    cols=frame.columns
    cols=pd.Index.tolist(cols)
    cols=pd.Series(cols)
    feats=pd.Series(['prev_str','prev_mi','prev_pad','prev_dm','prev_ht','prev_af','prev_hc','prev_stra','premrs', 	#history
'med_apt','med_noa','med_cou','med_hep','med_hdu','med_statin', 	#medication
'ct_bl_has','ct_bl_is','ct_bl_ht','ct_bl_leuk','ct_bl_leukd','ct_bl_leukloc','ct_bl_old','ct_bl_aspectsold_nc', 	#ct blanco
'ct_bl_aspects_nc','ct_bl_aspects_nl','ct_bl_aspects_ci','ct_bl_aspects_ins','ct_bl_aspects_m1','ct_bl_aspects_m2','ct_bl_aspects_m3',
'ct_bl_aspects_m4','ct_bl_aspects_m5','ct_bl_aspects_m6', 	#ct blanco aspects
'ct_bl_aspectsold_nl','ct_bl_aspectsold_ci','ct_bl_aspectsold_ins','ct_bl_aspectsold_m1','ct_bl_aspectsold_m2','ct_bl_aspectsold_m3',
	'ct_bl_aspectsold_m4','ct_bl_aspectsold_m5','ct_bl_aspectsold_m6', 	#ct blanco aspects 'old'
'intracranath_c','intracranvma_c','occlsegment_c','secoccl_c','secocclloc_c','collaterals','otherocclsegmangio_c','cbs_occlsegment_recoded'
,'A1hypoplasia','A2hypoplasia','collaterals_ex', 	#cta
'cta_cbs_ica','cta_cbs_sica','cta_cbs_a1','cta_cbs_m1prox','cta_cbs_m1dis','cta_cbs_m2temp','cta_cbs_m2fp','cta_cbs_a2',	#cbs baseline
'nihssbl_loc','nihssbl_q','nihssbl_com','nihssbl_gaze','nihssbl_vis','nihssbl_fac','nihssbl_larm','nihssbl_rarm','nihssbl_lleg','nihssbl_rleg',
	'nihssbl_ata','nihssbl_sen','nihssbl_afa','nihssbl_dys','nihssbl_ex',	#nihss baseline
'glucose','rr_syst','rr_dias','height','temp','weight','INR','trombo','creat','crp',	 #vital/lab
'age','sex','fudate','smoking','wkend','offhours','refhospital','ivtrom','ivtdose','ivtci','inhosp','gcse','gcsm','gcsv', 	#misc
'togroin' 	#timing
     ])
    
    final_feats=feats[feats.isin(cols)]
            
    frame=frame[final_feats]
    vals_mask=['premrs','collaterals','smoking']
    return(frame,vals_mask)

#one-hot-encode: test first
    """
    mask=['premrs','collaterals','collaterals_ex','pretici_c','occlsegment_c','occlsegmangio_c','smoking']
    mask_frame=frame[mask]
    mask_frame=np.array(mask_frame,dtype='float64')
    rf_enc = OneHotEncoder()
    rf_enc.fit(mask_frame,y=none)
    Result=rf_enc.transform(X_vars)
    Result=Result.toarray()
    Result=np.array(Result,dtype='float64')
    new_frame=frame.drop(mask_frame,axis=1)
    new_frame=np.array(new_frame)
    new_frame = np.concat(new_frame,Result,axis=1)
    """
#missing data: test first


def Get_Vars_Baseline_contscores(frame):  
    cols=frame.columns
    cols=pd.Index.tolist(cols)
    cols=pd.Series(cols)
    feats=pd.Series(['prev_str','prev_mi','prev_pad','prev_dm','prev_ht','prev_af','prev_hc','prev_stra','premrs', 	#history
'med_apt','med_noa','med_cou','med_hep','med_hdu','med_statin', 	#medication
'ct_bl_has','ct_bl_is','ct_bl_ht','ct_bl_leuk','ct_bl_leukd','ct_bl_leukloc','ct_bl_old','ct_bl_aspectsold_nc', 	#ct blanco
'ASPECTS_BL',#ct blanco aspects
'ASPECTS_BLold',#ct blanco aspects 'old'
'intracranath_c','intracranvma_c','occlsegment_c','secoccl_c','secocclloc_c','collaterals',
	'cbs_occlsegment_recoded','A1hypoplasia','A2hypoplasia','collaterals_ex', 	#cta
'cta_cbs_ica','cta_cbs_sica','cta_cbs_a1','cta_cbs_m1prox','cta_cbs_m1dis','cta_cbs_m2temp','cta_cbs_m2fp','cta_cbs_a2',	#cbs baseline
'NIHSS_BL',#nihss baseline
'glucose','rr_syst','rr_dias','height','temp','weight','INR','trombo','creat','crp',	 #vital/lab
'age','sex','fudate','smoking','wkend','offhours','refhospital','ivtrom','ivtdose','ivtci','inhosp','gcs',	#misc
'togroin', 	#timing
     ])
    
    final_feats=feats[feats.isin(cols)]
            
    frame=frame[final_feats]
    vals_mask=['premrs','collaterals','smoking']
    return(frame,vals_mask)

#dichotomise
#'premrs','collaterals','pretici_c','occlsegment_c','occlsegmangio_c','smoking'

 
def Get_Vars_All_binscores(frame):  
    cols=frame.columns
    cols=pd.Index.tolist(cols)
    cols=pd.Series(cols)
    feats=pd.Series(['prev_str','prev_mi','prev_pad','prev_dm','prev_ht','prev_af','prev_hc','prev_stra','premrs', 	#history
'med_apt','med_noa','med_cou','med_hep','med_hdu','med_statin', 	#medication
'ct_bl_has','ct_bl_is','ct_bl_ht','ct_bl_leuk','ct_bl_leukd','ct_bl_leukloc','ct_bl_old','ct_bl_aspectsold_nc', 	#ct blanco
'ct_bl_aspects_nc','ct_bl_aspects_nl','ct_bl_aspects_ci','ct_bl_aspects_ins','ct_bl_aspects_m1','ct_bl_aspects_m2','ct_bl_aspects_m3',
	'ct_bl_aspects_m4','ct_bl_aspects_m5','ct_bl_aspects_m6', 	#ct blanco aspects
'ct_bl_aspectsold_nl','ct_bl_aspectsold_ci','ct_bl_aspectsold_ins','ct_bl_aspectsold_m1','ct_bl_aspectsold_m2','ct_bl_aspectsold_m3',
	'ct_bl_aspectsold_m4','ct_bl_aspectsold_m5','ct_bl_aspectsold_m6', 	#ct blanco aspects 'old'
'intracranath_c','intracranvma_c','occlsegment_c','occlsegmangio_c','secoccl_c','secocclloc_c','collaterals','otherocclsegmangio_c',
	'cbs_occlsegment_recoded','A1hypoplasia','A2hypoplasia','collaterals_ex',    	#cta
'cta_cbs_ica','cta_cbs_sica','cta_cbs_a1','cta_cbs_m1prox','cta_cbs_m1dis','cta_cbs_m2temp','cta_cbs_m2fp','cta_cbs_a2',	#cbs baseline
'pretici_c','diss_c','INT_c','perf_c','disttrom_c','spasm_c','dsaich_c','othercom_c','occlsideangio_c','preaol_c',#dsa
'nihssbl_loc','nihssbl_q','nihssbl_com','nihssbl_gaze','nihssbl_vis','nihssbl_fac','nihssbl_larm','nihssbl_rarm','nihssbl_lleg','nihssbl_rleg',
	'nihssbl_ata','nihssbl_sen','nihssbl_afa','nihssbl_dys','nihssbl_ex',	#nihss baseline
'glucose','rr_syst','rr_dias','height','temp','weight','INR','trombo','creat','crp',	 #vital/lab
'age','sex','fudate','smoking','cnonr','wkend','offhours','refhospital','ivtrom','ivtdose','ivtci','inhosp','gcse','gcsm','gcsv', 	#misc
'togroin', 	#timing
'balloon','interv_comp','heparin','heparindose','performedproc','iatreatment1','stentother1','asp1','merci1_attempts','otherdevice1','iaurodose1',
	'iaaltdose1','iatreatment2','stentother2','asp2','otherdevice2','iaurodose2','iaaltdose2','iatreatment3','stentother3','asp3','otherdevice3',
	'iaurodose3','reopro','reopro_dose','anymed','hemicraniectomy','stenttotal','iauro1','iauro2','iauro3','iaalt1','iaalt2','iaalt3','total_attempts',
	'postaol_c','icastent_c','balloonplast_c','vascinj_c', 	#interv
'onset_reperfusion_c','dtnt','durproc',		#timing fu
'GA','CS','ic_stay','ic_days','hc_stay','hc_days','sc_stay','sc_days','disloc','NIHSSPI_BL','neurodetnihss','success_rep',	#misc fu
'sae_PS','sae_IS','sich_c','sae_ech','sae_ci','sae_ar','sae_pneum','sae_inf','saedate1','saedate2','saedate3',
	'saedate4','saedate5',		#adverse events
'nihsspi_loc','nihsspi_q','nihsspi_com','nihsspi_gaze','nihsspi_vis','nihsspi_fac','nihsspi_larm','nihsspi_lleg','nihsspi_rleg',
	'nihsspi_ata','nihsspi_sen','nihsspi_afa','nihsspi_dys','nihsspi_ex',	  #nihss fu
'ichtype3_c','ichtype4_c','hemicran_c','shift_c','ich_c','ichtype1_c','ichtype2_c','ichrelation_c',	#cta follow-up
'ct_fu_aspects_nc','ct_fu_aspects_nl','ct_fu_aspects_ci','ct_fu_aspects_ins','ct_fu_aspects_m1','ct_fu_aspects_m2','ct_fu_aspects_m3',
	'ct_fu_aspects_m4','ct_fu_aspects_m5','ct_fu_aspects_m6',  #cta aspects follow-up
   ])
    
    final_feats=feats[feats.isin(cols)]
            
    frame=frame[final_feats]
    vals_mask=['premrs','collaterals','pretici_c','occlsegment_c','occlsegmangio_c','smoking','performedproc','disloc']
    return(frame,vals_mask)    
#dichotomise
#'premrs','collaterals','pretici_c','occlsegment_c','occlsegmangio_c','smoking','performedproc','disloc'

 #This was created to add postici and cluster all adverse events into any     
def Get_Vars_All_contscores(frame):  
    cols=frame.columns
    cols=pd.Index.tolist(cols)
    cols=pd.Series(cols)
    feats=pd.Series(['prev_str','prev_mi','prev_pad','prev_dm','prev_ht','prev_af','prev_hc','prev_stra','premrs',
'med_apt','med_noa','med_cou','med_hep','med_hdu','med_statin',
'ct_bl_has','ct_bl_is','ct_bl_ht','ct_bl_leuk','ct_bl_leukd','ct_bl_leukloc','ct_bl_old','ct_bl_aspectsold_nc', 	#ct blanco
'ASPECTS_BL',#ct blanco aspects
'ASPECTS_BLold',#ct blanco aspects 'old'
'intracranath_c','intracranvma_c','occlsegment_c','occlsegmangio_c','secoccl_c','secocclloc_c','collaterals','otherocclsegmangio_c',
	'cbs_occlsegment_recoded','A1hypoplasia','A2hypoplasia','collaterals_ex', 	#cta
'cta_cbs_ica','cta_cbs_sica','cta_cbs_a1','cta_cbs_m1prox','cta_cbs_m1dis','cta_cbs_m2temp','cta_cbs_m2fp','cta_cbs_a2',	#cbs baseline
'pretici_c','diss_c','INT_c','perf_c','disttrom_c','spasm_c','dsaich_c','othercom_c','occlsideangio_c','preaol_c',
#dsa
'NIHSS_BL',#nihss baseline
'glucose','rr_syst','rr_dias','height','temp','weight','INR','trombo','creat','crp',	 #vital/lab
'age','sex','fudate','smoking','wkend','offhours','refhospital','ivtrom','ivtdose','ivtci','inhosp','gcs',	#misc
'togroin', 	#timing
            'balloon','interv_comp','heparin','heparindose','performedproc','iatreatment1','stentother1','asp1','merci1_attempts','otherdevice1','iaurodose1',
	'iaaltdose1','iatreatment2','stentother2','asp2','otherdevice2','iaurodose2','iaaltdose2','iatreatment3','stentother3','asp3','otherdevice3',
	'iaurodose3','reopro','reopro_dose','anymed','hemicraniectomy','stenttotal','iauro1','iauro2','iauro3','iaalt1','iaalt2','iaalt3','total_attempts',
	'postaol_c','icastent_c','balloonplast_c','vascinj_c', 	#interv
'onset_reperfusion_c','dtnt','durproc',		#timing fu
'GA','CS','ic_stay','ic_days','hc_stay','hc_days','sc_stay','sc_days','disloc','NIHSSPI_BL','neurodetnihss','success_rep',	#misc fu
'sae_PS','sae_IS','sich_c','sae_ech','sae_ci','sae_ar','sae_pneum','sae_inf','saedate1','saedate2','saedate3',
	'saedate4','saedate5',		#adverse events
'NIHSS_FU',  #nihss fu
'ichtype3_c','ichtype4_c','hemicran_c','shift_c','ich_c','ichtype1_c','ichtype2_c','ichrelation_c',	#cta follow-up
'ASPECTS_FU'   #cta aspects follow-up
])
    final_feats=feats[feats.isin(cols)]            
    frame=frame[final_feats]
    vals_mask=['premrs','collaterals','pretici_c','occlsegment_c','occlsegmangio_c','smoking','performedproc','disloc']
    return(frame,vals_mask)    

    return(frame)  
#dichotomise
#'premrs','collaterals','pretici_c','occlsegment_c','occlsegmangio_c','smoking','performedproc','disloc'



def Get_Vars_priorknowledge_baseline(frame):  
    cols=frame.columns
    cols=pd.Index.tolist(cols)
    cols=pd.Series(cols)
    feats=pd.Series(['age','NIHSS_BL','prev_dm','prev_str','prev_af','premrs','rr_syst','ivtrom','togroin','ASPECTS_BL','occlsegmangio_c','collaterals'])
    final_feats=feats[feats.isin(cols)]            
    frame=frame[final_feats]
    vals_mask=['premrs','collaterals','occlsegmangio_c']
    return(frame,vals_mask)
#dichotomise
#'premrs','collaterals','pretici_c','occlsegmangio_c','smoking','collaterals_ex'

def Get_Vars_priorknowledge_all(frame): 
    cols=frame.columns
    cols=pd.Index.tolist(cols)
    cols=pd.Series(cols)
    feats=pd.Series(['age','premrs','prev_dm','prev_ht','prev_str','rr_syst','ivtrom','collaterals','togroin','occlsegmangio_c',
                 'GA','sich_c','NIHSS_FU','posttici_c','durproc'])
    final_feats=feats[feats.isin(cols)]            
    frame=frame[final_feats]
    vals_mask=['premrs','collaterals','occlsegmangio_c']
    return(frame,vals_mask)
#dichotomise
#'premrs','collaterals','pretici_c','occlsegment_c_short','smoking','collaterals_ex'


    
def Change2_Missing_spss(data,cols): 
    """
    In the spss file many features have different values for missing, like 2 instead of np.nan, here we change those
    Input = frame with wrong missing values
    Output = Fixed frame
    
    """
    cols=pd.Index.tolist(cols)
    pos=(int)(cols.index('ivtrom'))
    
    print(data.shape)
    print(pos)
    print(data[0,0]>2)
    for i in range(0,data.shape[0]):
        if data[i,pos]>=2:
            data=np.nan
    return(data)   
    
    
def Impute_and_Save(f):
    raw_data_list = list(f) 
    frame = pd.DataFrame(raw_data_list) 
    frame = frame.rename(columns=frame.loc[0]).iloc[1:] 
    #frame=frame.drop([b'StudySubjectID'],axis=1)
    cols=frame.columns #these columns are in a binary format, below they are converted to string
    colsaux=[]
    
    for i in range(0,cols.shape[0]):
        colsaux.append(cols[i].decode('UTF8'))

    cols=colsaux    
    frame.columns=cols
    
    arr=np.array(frame['StudySubjectID'])
    for i in range(0,arr.shape[0]):
        arr[i]=arr[i].decode('UTF8')
    np.save('E:\\Adam\\sub_id_complete.npy',arr)
    
    frame.fillna(value=np.nan,inplace=True)
    
    Ys=(frame[['mrs','posttici_c']]).values
    #Ys=(frame[['mrs']]).values
    cont_mis=0
    for i in range(0,frame.shape[0]):
        if (np.isnan(Ys[i,0]) or np.isnan(Ys[i,1])):
            print('missing:', arr[i])
            frame=frame.drop([i+1])
            cont_mis=cont_mis+1 
            
    print(cont_mis)
    arr=np.array(frame['StudySubjectID'])
    for i in range(0,arr.shape[0]):
        arr[i]=arr[i].decode('UTF8')
    np.save('E:\\Adam\\sub_id.npy',arr)
    return(arr)
    
    """
    #cols=frame.columns
    Ys_final=(frame[['mrs','posttici_c']]).values
    
    frame=frame.drop(['mrs','posttici_c'],axis=1)
    
    cols=frame.columns
    for i in range (0,frame.shape[1]):
        col=frame.columns[i]
        val=frame[col]
        s=val.dtype
        if s=='object':
            frame[col]=pd.to_numeric(frame[col],errors='coerce')
          
    
    dataread=np.array(frame.values,dtype='float64')    
    num_miss=np.zeros(dataread.shape[1])
    
    for i in range(0,dataread.shape[0]):
        for j in range(0,dataread.shape[1]):
            if np.isnan(dataread[i,j]):
                num_miss[j]=num_miss[j]+1
    #how much% is missing, 25% or more
    for i in range(0,dataread.shape[1]):
        num_miss[i]=(num_miss[i]*100)/dataread.shape[0]
        
    #check if below <25 because I want the above 25% to be 0 and eliminated 

    
    cols_delete=num_miss<25
    for i in range(0,len(cols)):
        if cols_delete[i]==1:
            print(cols[i],num_miss[i])
        
    dataread=dataread[:,cols_delete]
    
    
    cols=cols[cols_delete]    
   
    mice=MICE()
    X=mice.complete(dataread)
    cols_ind=pd.Index.tolist(cols)

    
    #df=pd.DataFrame(X,columns=cols_ind)
    
    df=pd.DataFrame(X)
    
    df=df.round()
    df_new=df
    
    cols_ind.append('mrs')
    cols_ind.append('posttici_c')
    #df_new=pd.concat([df_new,Ys_final],axis=1)
    
    X=np.array(df_new,dtype='float64') 
    X=np.concatenate((X,Ys_final),axis=1)
    
    df_new=pd.DataFrame(X,columns=cols_ind)
    
    return(X,cols,df_new)
    """
 
def Fix_Dataset_spss(f,label_name,feats_use,binary_mrs):
    """
    This function reads the dataset in a spss format, selected only the important collumns,
    preprocess a few of them into cathegories and performs imputation using random forests or MICE
    Input:
    f = returned from spss.SavReader(filename)
    label_name = name of the column to be predicted, the label for the features (Y)
    Feats_use = This parameter specifies which variables will be selected ('Baseline_Imp','Baseline_NonImp','ALL_NonImp')
    binary_mrs= If true returns a binary version of mrs >2 =1 and <=2 =0, if false returns it from 1 to 6 (multiclass)
    Output:
        X = Dataset features with imputed values (mxn)
        Y = Labels (m)
        cols = columns names so one can trace back each feature

    """

                
    raw_data_list = list(f) 
    frame = pd.DataFrame(raw_data_list) 
    frame = frame.rename(columns=frame.loc[0]).iloc[1:] 
    original_frame=frame
    cols=frame.columns #these columns are in a binary format, below they are converted to string
    colsaux=[]
    
    for i in range(0,cols.shape[0]):
        colsaux.append(cols[i].decode('UTF8'))
    cols=colsaux    
    frame.columns=cols


    frame.fillna(value=np.nan,inplace=True)
    
    #Patients with missing mrs or postici are deleted    
    Ys=(frame[['mrs','posttici_c']]).values    
    cont_mis=0
    for i in range(0,frame.shape[0]):
        if (np.isnan(Ys[i,0]) or np.isnan(Ys[i,1])):
            #print("deleting ",Ys_t.loc[i+1,['mrs']],Ys_t.loc[i+1,['posttici_c']])
            frame=frame.drop([i+1])
            cont_mis=cont_mis+1 
    

    Y=frame[label_name]
    Y=np.array(Y,dtype='int32')

    print(cont_mis)
    if binary_mrs:
        for i in range(0,Y.shape[0]):
            if Y[i]>2:
                Y[i]=1
            else:
                Y[i]=0
    
    frame=frame.drop(label_name,axis=1)  

    #checking what kind of features will be used based on the experiment set up
    """
    frame = {
          'Baseline_Imp': lambda frame:Get_Vars_Baseline(frame),
          'ALL_NonImp': lambda frame:Get_Vars_AllNonImp(frame),
          'Knowledge':lambda frame:Get_Vars_Know(frame),
          'Raw_Baseline_Imp': lambda frame:Get_Vars_Baseline_raw(frame),
          'RawContinuous_Baseline_Imp': lambda frame:Get_Vars_Baseline_ContinuousScores(frame),
          'Raw_All_Vars': lambda frame:Get_Vars_All_raw(frame),
          'Raw_Continuous_All_Vars': lambda frame:Get_Vars_All_ContinuousScores(frame),
    }[feats_use](frame)
    """
    for i in range (0,frame.shape[1]):
        col=frame.columns[i]
        val=frame[col]
        s=val.dtype
        if s=='object':
            frame[col]=pd.to_numeric(frame[col],errors='coerce')
        
       
    cols=frame.columns
    
    dataread=np.array(frame.values,dtype='float64')    
    num_miss=np.zeros(dataread.shape[1])
    
    for i in range(0,dataread.shape[0]):
        for j in range(0,dataread.shape[1]):
            if np.isnan(dataread[i,j]):
                num_miss[j]=num_miss[j]+1
    #how much% is missing, 25% or more
    for i in range(0,dataread.shape[1]):
        num_miss[i]=(num_miss[i]*100)/dataread.shape[0]
        
    #check if below <25 because I want the above 25% to be 0 and eliminated 
    cols_delete=num_miss<25
    
    #for j in range(0,dataread.shape[1]):
    #    if num_miss[j]>1000:
    #        print(cols[j])
    
    dataread=dataread[:,cols_delete]
    cols=cols[cols_delete]
    #X=imp.IARI(dataread,Y)

    #frame=Change2_Missing_spss(dataread,cols)        
    
    
    mice=MICE()
    X=mice.complete(dataread)

    df=pd.DataFrame(X,columns=cols)
    
    df=df.round()
    df_new=df
    
    
    X=np.array(df_new,dtype='float64') 
    
    return(X,Y,cols,original_frame)

def Fix_Dataset_csv(f,feats_use):
    """
    This function reads the dataset in a spss format, selected only the important collumns,
    preprocess a few of them into cathegories and performs imputation using random forests or MICE
    Input:
    
    label_name = name of the column to be predicted, the label for the features (Y)
    Feats_use = This parameter specifies which variables will be selected ('Baseline_Imp','Baseline_NonImp','ALL_NonImp')
    binary_mrs= If true returns a binary version of mrs >2 =1 and <=2 =0, if false returns it from 1 to 6 (multiclass)
    Output:
        X = Dataset features with imputed values (mxn)
        Y = Labels (m)
        cols = columns names so one can trace back each feature

    """

    frame=pd.read_csv(f)
    center=frame['cnonr'].values
    cols=frame.columns #these columns are in a binary format, below they are converted to string
    
    #checking what kind of features will be used based on the experiment set up
    
    frame,vals_mask = {
          'Baseline_binscore': lambda frame:Get_Vars_Baseline_binscores(frame),
          'Baseline_contscore': lambda frame:Get_Vars_Baseline_contscores(frame),
          'All_vars_binscore':lambda frame:Get_Vars_All_binscores(frame),
          'All_vars_contscore': lambda frame:Get_Vars_All_contscores(frame),
          'Knowledge_baseline': lambda frame:Get_Vars_priorknowledge_baseline(frame),
          'Knowledge_all': lambda frame:Get_Vars_priorknowledge_all(frame),
    }[feats_use](frame)

    cols=frame.columns
    
    dataread=np.array(frame)

    return(dataread,cols,center,vals_mask)

  