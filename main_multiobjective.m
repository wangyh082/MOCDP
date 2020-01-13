clear ; close all; clc
addpath('data\')
% Load Training Data
fprintf('Loading Data\n');
% 399*3���Լ�

    CR = 0.2;
    for num=1:1:1
        switch num
            case 1
                load('Alizadeh-2000-v1.mat');
        end
        %fea = normalizeData(fea);
        data=[fea gnd];
        cluster_num = length(unique(gnd));
        times=1;
        rand('state', 0);
        for runs=1:1:times
            disp(runs)
            A = 1;
            
            isPlot = true;
            isKernal = false;
            ND = max(data(:,2));
            NL = max(data(:,1));
            if (NL>ND)
                ND = NL;   %% ȷ�� DN ȡΪ��һ�������ֵ�еĽϴ��ߣ���������Ϊ���ݵ�����
            end
            Dist=pdist2(data(:,1:end-1),data(:,1:end-1));
            %% data ��һ��ά�ȵĳ��ȣ��൱���ļ�����������������ܸ�����
            N = size(data,1);
            %% ��ʼ��Ϊ��
            c=zeros(N,N);
            NumD=size(data,2)-1;
            ObjectiveNum=2;
            H = [99 13  7  5  4  0  3  0  2];
            H = H(ObjectiveNum-1);
            [popsize,W] = F_weight(H,ObjectiveNum);
            W(W==0) = 0.000001;
            T = 2;
            %�ھ��ж�
            BX = zeros(popsize);
            for i = 1 : popsize
                for j = i : popsize
                    BX(i,j) = norm(W(i,:)-W(j,:));
                    BX(j,i) = BX(i,j);
                end
            end
            [~,BX] = sort(BX,2);
            BX = BX(:,1:T);
            Population=rand(popsize,NumD+1);
            Sum_pop=sum(Population(:,1:end-1)')';
            repmat(Sum_pop,1,NumD);
            Population1=Population(:,1:end-1)./repmat(Sum_pop,1,NumD);
            for qkt=1:1:popsize
                for i = 1:N-1
                    ix = data(i,1:end-1);
                    for  j=i+1:N;
                        jx = data(j,1:end-1);
                        c(i,j) = sqrt(sum(Population1(qkt,:).*(ix-jx).^2));
                        c(j,i) =  c(i,j);
                    end
                end
                NX = size(c, 1) * (size(c, 2) - 1) / 2;
                ND = size(c, 1);
                percent = Population(qkt,NumD+1);
                position = floor(NX*percent+1);  %% round ��һ���������뺯��
                sda = sort(squareform(c));
                dc = sda(position);
                
                
                rho = [];
                for i = 1:ND
                    rho(i) = 0.;
                end
                
                rho = sum((c-dc)<0, 2);
                
                %rho=rho';
                %% ������������ֵ���������ֵ�����õ����о���ֵ�е����ֵ
                if isKernal == true
                    maxd=max(max(dist));
                else
                    maxd=max(max(c));
                end
                %% �� rho ���������У�ordrho ������
                [rho_sorted, ordrho]=sort(rho,'descend');
                
                %% ���� rho ֵ�������ݵ�
                delta(ordrho(1))=-1.;
                nneigh(ordrho(1))=0;
                
                %% ���� delta �� nneigh ����
                if isKernal == true
                    for ii=2:ND
                        delta(ordrho(ii))=maxd;
                        for jj=1:ii-1
                            if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
                                delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
                                nneigh(ordrho(ii))=ordrho(jj);
                            end
                        end
                    end
                else
                    for ii=2:ND
                        delta(ordrho(ii))=maxd;
                        for jj=1:ii-1
                            if(c(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
                                delta(ordrho(ii))=c(ordrho(ii),ordrho(jj));
                                nneigh(ordrho(ii))=ordrho(jj);
                            end
                        end
                    end
                end
                %% ���� rho ֵ������ݵ�� delta ֵ
                delta(ordrho(1))=max(delta(:));
                %% ind �� gamma �ں��沢û���õ�
                for i=1:ND
                    ind(i)=i;
                    gamma(i)=rho(i)*delta(i);
                end
                
                NCLUST = cluster_num;
                for i=1:ND
                    cl(i)=-1;
                end
                [B, Index] = sort(gamma, 'descend');
                % cl��ÿ�����ݵ����������
                % icl�����о������ĵ����
                icl = Index(1:NCLUST);
                cl(Index(1:NCLUST)) = 1:NCLUST;
                %% �ھ���������ͳ�����ݵ㣨���������ģ��ĸ���
                mm=data(icl,1:end-1);
                % Calculate Distance Matrix
                for i=1:ND
                    if (cl(ordrho(i))==-1)
                        cl(ordrho(i))=cl(nneigh(ordrho(i)));
                    end
                end
                dtype=1;
                [DB,CH,Dunn,KL,Han,~] = valid_internal_deviation(data(:,1:end-1),cl,dtype);
                cp = valid_compactness(data(:,1:end-1), cl);
                Objective_Value3=0;
                %Objective_Value4=Edge(cl,data,c);
                FunctionValue(qkt,:)=[cp -CH];
                label_FunctionValue(qkt,:)=cl;
                % rrs(qkt,:)=Objective_Value3;
                
            end
            Boundary=[1;0];
            Coding='Real';
            Z = min(FunctionValue);
            generation=1;
            for tt=1:1:generation;
                numberofupdate = 0;
                for ip = 1 : popsize
                    P = 1:popsize;
                    kx = randperm(length(P));
                    %�����Ӵ�                    
                    Offspring = F_generator(Population(ip,:),Population(P(kx(1)),:),Population(P(kx(2)),:),Population(P(kx(3)),:),Population(P(kx(4)),:),Boundary,CR);
                    Offspring1=Offspring(1:end-1)./sum(Offspring(1:end-1));
                    %Offspring1=Offspring;
                    for i = 1:N-1
                        ix = data(i,1:end-1);
                        for  j=i+1:N;
                            jx = data(j,1:end-1);
                            c(i,j) = sqrt(sum(Offspring(1:end-1).*(ix-jx).^2));
                            c(j,i) =  c(i,j);
                        end
                    end
                    percent = Offspring(end);
                    position = floor(NX*percent+1);  %% round ��һ���������뺯��
                    sda = sort(squareform(c));
                    dc = sda(position);
                    rho = [];
                    for i = 1:ND
                        rho(i) = 0.;
                    end
                    
                    rho = sum((c-dc)<0, 2);
                    maxd=max(max(c));
                    [rho_sorted, ordrho]=sort(rho,'descend');
                    %% ���� rho ֵ�������ݵ�
                    delta(ordrho(1))=-1.;
                    nneigh(ordrho(1))=0;
                    for ii=2:ND
                        delta(ordrho(ii))=maxd;
                        for jj=1:ii-1
                            if(c(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
                                delta(ordrho(ii))=c(ordrho(ii),ordrho(jj));
                                nneigh(ordrho(ii))=ordrho(jj);
                            end
                        end
                    end
                    delta(ordrho(1))=max(delta(:));
                    for i=1:ND
                        ind(i)=i;
                        gamma(i)=rho(i)*delta(i);
                    end
                    NCLUST = cluster_num;
                    %% cl Ϊ������־���飬cl(i)=j ��ʾ�� i �����ݵ�����ڵ� j �� cluster
                    %% ��ͳһ�� cl ��ʼ��Ϊ -1
                    for i=1:ND
                        cl(i)=-1;
                    end
                    [BV, Index] = sort(gamma, 'descend');
                    % cl��ÿ�����ݵ����������
                    % icl�����о������ĵ����
                    icl = Index(1:NCLUST);
                    cl(Index(1:NCLUST)) = 1:NCLUST;
                    for i=1:ND
                        if (cl(ordrho(i))==-1)
                            cl(ordrho(i))=cl(nneigh(ordrho(i)));
                        end
                    end
                    label_OffFunValue=cl;
                    [DB,CH,Dunn,KL,Han,~] = valid_internal_deviation(data(:,1:end-1),cl,dtype);
                    cp = valid_compactness(data(:,1:end-1), cl);
                    OffFunValue=[cp -CH];
                    %OffFunValue = (OffFunValue-Fmin)./(Fmax-Fmin);
                    %�������������
                    Z = min(Z,OffFunValue);
                    
                    %�����ھӸ���
                    for j = 1 : T
                        if A == 1
                            g_old = max(abs(FunctionValue(BX(ip,j),:)-Z).*W(BX(ip,j),:));
                            g_new = max(abs(OffFunValue-Z).*W(BX(ip,j),:));
                        elseif A == 2
                            d1 = abs(sum((FunctionValue(BX(ip,j),:)-Z).*W(BX(ip,j),:)))/norm(W(BX(ip,j),:));
                            g_old = d1+5*norm(FunctionValue(BX(ip,j),:)-(Z+d1*W(BX(ip,j),:)/norm(W(BX(ip,j),:))));
                            d1 = abs(sum((OffFunValue-Z).*W(BX(ip,j),:)))/norm(W(BX(ip,j),:));
                            g_new = d1+5*norm(OffFunValue-(Z+d1*W(BX(ip,j),:)/norm(W(BX(ip,j),:))));
                        end
                        if g_new < g_old
                            %���µ�ǰ�����ĸ���
                            numberofupdate = numberofupdate + 1;
                            Population(BX(ip,j),:) = Offspring;
                            FunctionValue(BX(ip,j),:) = OffFunValue;
                            label_FunctionValue(BX(ip,j),:)=label_OffFunValue;
                            %                 rrs(BX(ip,j),:)=rrx;
                        end
                    end
                    %FunctionValue = FunctionValue.*repmat(Fmax-Fmin,popsize,1)+repmat(Fmin,popsize,1);
                end
                label_FunctionValue;
                %
                for ij=1:1:popsize
                    [~, ~, Rn, NMI] = exMeasure(label_FunctionValue(ij,:)',data(:,end));
                    rr(ij,:)=[ NMI Rn];
                end           
            end
            [rrmin,Indd]=max(rr(:,1));          
        end
        finalresult(num,:)=rr(Indd,:)
        clearvars -except num finalresult XXX py CR
    end
%     dlmwrite(['data' num2str(CR) '.csv' ],finalresult, 'precision', 4, 'newline', 'pc');
% end



