function [P, C]=kmeans_act(S,C,P,D,iter)
    for i=1:iter
        [C, active, activeCount, Pact]=OptimalRepresentativesAct(S,C,P);
        [P, D]=OptimalPartitionAct(D,activeCount,active,S,C,P, Pact);   
        clear active activeCount;
    end
clear D;
end

%%===========================================
function [C, active, activeCount, P]=OptimalRepresentativesAct(S, C, P)
active=[];
activeCount=1;
% Update centroids using means of partitions
    for t=1:size(C,1);
        v=C(t,:);
        members = (P == t);
        if any(members)
            C(t,:)=sum(S(members,:),1)/sum(members);
            if CompareVectors(C(t,:), v)~=1
                active(activeCount)=t;
                P(members)=1;
                activeCount=activeCount+1;
            else
                P(members)=0;
            end
        else 
             warning('random centre created');
             k=ceil(rand(1) *size(S,1));
             C(t,:) = S(k,:);
             P(k)=1;
        end

        clear members;
    end
activeCount=activeCount-1;
% P now contains 1's where point is in active centroid and 0's otherwise
end

function [P, D]=OptimalPartitionAct(D, activeCount, active, S, C, P, Pact)
if activeCount>0  
     Cact=C(active,:); % active codebook
     %= Works with points in static clusters =%
     Sn=S(Pact==0,:); 
     if size(Sn,1)>0
         Pn=P(Pact==0); 
         [near, error] = FindNearestVectors(Sn, Cact);
         near=active(near);
         ind=D(Pact==0)<error; % =1 if actual centroid is closer than others
         if sum(ind)>0
            near(ind(ind==1))=Pn(ind==1);
            error(ind(ind==1))=D(Pn(ind==1));
         end
         D(Pact==0)=error;
         P(Pact==0)=near;
     end
     clear ind Sn;
     
     %= Works with points in active clusters =%
     Sact=S(Pact==1,:); 
     PP=P.*Pact;
     % Compare distances to new centroids and old centroids
     dis=CalculateDistances(Sact, Cact, active, PP(PP>0), size(S,1)); 
     clear Sact;
     ind=dis<=D';
     P1=Pact&ind';
     P2=xor(Pact,ind');
     clear Pact ind;
     %= Works with points in active clusters, centroid moves closer =%
     if sum(P1)>0
        Sa=S(P1,:);
         [near, error] = FindNearestVectors(Sa, Cact); % search subcodebook
         near=active(near);
         P(P1)=near;
         D(P1)=error;
     end
     clear P1;
     %= Works with points in active clusters, centroid moves farther =%
     if sum(P2)>0
         Sa=S(P2,:);
         [near, error] = FindNearestVectors(Sa, C); % search codebook
         P(P2)=near;
         D(P2)=error;
     end
     clear P2;
else
     return
end
end

%===========================================
% Internal functions 
function D=CalculateDistances(S, C, active, P, l)
% Calculates distances from each data point to its centroid
D=inf(l,1);
    for i=1:size(C,1)
        k=active(i); % centroid prosessed
        s=S(P==k,:);
        dis = zeros(1,size(s,1));  
        dis(1,:)=VectorDistance(s, C(i,:));
        ind=P==k;
        D(ind)=dis;
        clear ind;
    end    
end

function dis=VectorDistance(v1, v2)
% Finds eucledean distance between two vectors
dis=0;
    for i=1:size(v1,2)
        dis = dis+ (v1(:,i) - v2(1,i)).^2;
    end
end

function [minInds, dist] = FindNearestVectors(S, C)
c=size(C,1);
s=size(S,1); 
e=zeros(c,s);  
    for i=1:size(C,1)
        e(i,:)=VectorDistance(S, C(i,:));
    end
[dist, minInds]=min(e,[],1);
end

function r=CompareVectors(v1, v2)
ind=sum(v1 ~= v2); % zero if are indentical
    if ind==0
        r=1;
    else
        r=0;
    end
end

