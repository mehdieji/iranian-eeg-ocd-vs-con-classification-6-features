clear;
clc;

num_ch=18;
num_FCs=[];
for i=1:num_ch-1
    D=(i-1)*num_ch+i+1:i*num_ch;
    num_FCs=[num_FCs,D];
end

Num1=setdiff(1:num_ch*num_ch,num_FCs);

load('E:\Temporary\DropBox\Healthy Biomarkers\Data\OCD_Normal\Results\ttest P_Values\MIZ\wPLI')
for i=1:9
    PP=squeeze(P(:,:,i));
    PP(Num1)=1;
    P(:,:,i)=PP;
    
    TT=squeeze(T(:,:,i));
    TT(Num1)=0;
    T(:,:,i)=TT;
end

save('E:\Temporary\DropBox\Healthy Biomarkers\Data\OCD_Normal\Results\ttest P_Values\MIZ\wPLI','P','T')

load('E:\Temporary\DropBox\Healthy Biomarkers\Data\OCD_Normal\Results\ttest P_Values\MIZ\PLV')
for i=1:9
    PP=squeeze(P(:,:,i));
    PP(Num1)=1;
    P(:,:,i)=PP;
    
    TT=squeeze(T(:,:,i));
    TT(Num1)=0;
    T(:,:,i)=TT;
end

save('E:\Temporary\DropBox\Healthy Biomarkers\Data\OCD_Normal\Results\ttest P_Values\MIZ\PLV','P','T')

