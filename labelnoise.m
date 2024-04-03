function [TB] = labelnoise(p,T,c1,c2)
    TB=T;
    x1=find(T==c1);v1=length(x1);
    x2=find(T==c2);v2=length(x2);
    Q1=round(v1*p);
    Q2=round(v2*p);
    allInd = randperm(v1);
    allIND1B=allInd(1:Q1);
    TB(x1(allIND1B))=c2;
    allInd = randperm(v2);
    allIND2B=allInd(1:Q2);
    TB(x2(allIND2B))=c1;
end
