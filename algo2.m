function [Inl] = algo2(net,Sall,D,TauACC,label,SId)
        %%%SId: subjects ID
        Np=length(SId);
        %%%%%%%%%%%%%%%%%%%%%%%%
        y= net(D');
        tind = vec2ind(label');
        yind = vec2ind(y);
        %%RESULTS ON ALL THE DATSET%%%%%%%%%%%%%
        FP=(yind==2)&(tind==1);
        FN=(yind==1)&(tind==2);
        TP=(yind==2)&(tind==2);
        TN=(yind==1)&(tind==1);
        %%FP, FN, TP, TN computed FOR EACH PARTICIPANT%%%%%%%%%%%
        Sall_FP=Sall(FP);
        x=unique(Sall_FP);
        RESULTALLFP=zeros(1,Np);RESULTALLFN=zeros(1,Np);RESULTALLTN=zeros(1,Np);RESULTALLTP=zeros(1,Np);
        for k=1:length(x)
            RESULTALLFP(SId==x(k))=sum(Sall_FP==x(k));
        end
        Sall_FN=Sall(FN);
        x=unique(Sall_FN);
        for k=1:length(x)
            RESULTALLFN(SId==x(k))=sum(Sall_FN==x(k));
        end
        Sall_TN=Sall(TN);
        x=unique(Sall_TN);
        for k=1:length(x)
            RESULTALLTN(SId==x(k))=sum(Sall_TN==x(k));
        end
        Sall_TP=Sall(TP);
        x=unique(Sall_TP);
        for k=1:length(x)
            RESULTALLTP(SId==x(k))=sum(Sall_TP==x(k));
        end

        accuracy=(RESULTALLTN+RESULTALLTP)./(RESULTALLTN+RESULTALLTP+RESULTALLFP+RESULTALLFN);
        Inl=find(accuracy>TauACC);
end