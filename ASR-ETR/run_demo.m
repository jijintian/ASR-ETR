

clear;
clc
folder_now = pwd;
addpath([folder_now, '\funs']);
addpath([folder_now, '\dataset']);
dataname=["NGs","BBCSport"];
%% ==================== Load Datatset and Normalization ===================
for it_name = 1:length(dataname)
    load(strcat('dataset/',dataname(it_name),'.mat'));
    cls_num=length(unique(truelabel{1}));
    X=data';
    gt = truelabel{1};
    nV = length(X);
    for v=1:nV
        [X{v}]=NormalizeData(X{v});
    end
    
    %% ========================== Parameters Setting ==========================
    result=[];
    num = 0;
    max_val=0;
    record_num = 0;
    ii=0;
    anc =[2*cls_num,3*cls_num,4*cls_num,5*cls_num,6*cls_num,7*cls_num,8*cls_num];
    %% ============================ Optimization ==============================
    for i = -6:1:0
        for jj = -6:1:0
            for j = -4:1:0
                for k =1:7
                    alpha = 10^(i);
                    gamma = 10^(jj);
                    delta=10^(j);
                    anchor = anc(k);
                    ii=ii+1;
                    tic;
                    [y,U,Z,converge_Z,converge_Z_G] = Train_problem(X, cls_num, anchor,alpha,gamma,delta);
                    time = toc;
                    [result(ii,:),~]=  ClusteringMeasure(gt, y);
                    [result(ii,:),time]
                    if result(ii,1) > max_val
                        max_val = result(ii,1);
                        record = [i,jj,j,k,time];
                        record_result = result(ii,:);
                        record_c ={U,Z,converge_Z,converge_Z_G};
                        record_time = time;
                    end
                end
            end
        end
    end
save('.\result\result_ASR_ETR_'+dataname(it_name),'result','record','max_val','record_result','record_c','time')
end

