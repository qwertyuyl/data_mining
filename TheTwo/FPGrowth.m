%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML116
% Project Title: FP-Growth Algorithm for Association Rule Mining
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function out=FPGrowth(T,MST,MCT)

    % Step 1: Create Sorted Items List, in order of Descending Frequency
    Items=[];
    for i=1:numel(T)
        Items=union(Items,T{i});
    end
    Items=reshape(Items,1,[]);

    Count=zeros(size(Items));
    for i=1:numel(T)
        Count=Count+ismember(Items,T{i});
    end

    [~, SortOrder]=sort(Count,'descend');
    Items=Items(SortOrder);

    % Step 2: Create FP-Tree

    empty_node.Name=[];
    empty_node.Count=0;
    empty_node.Parent=[];
    empty_node.Children=[];
    empty_node.Path=[];
    empty_node.Patterns={};
    empty_node.PatternCount=[];

    % Add Base Node
    Node(1)=empty_node;
    Node(1).Name='n';
    Node(1).Parent=0;

    LastIndex=1;

    for i=1:numel(T)

        A=[];
        for item=Items
            if ismember(item,T{i})
                A=[A item];	%#ok
            end
        end

        CurrentNode=1;
        Node(CurrentNode).Count=Node(CurrentNode).Count+1;  %#ok

        for a=A

            ChildNodeExists=false;
            for c=Node(CurrentNode).Children
                if Node(c).Name==a
                    ChildNodeExists=true;
                    break;
                end
            end

            if ChildNodeExists
                CurrentNode=c;

            else
                NewNode=empty_node;
                NewNode.Name=a;
                NewNode.Parent=CurrentNode;
                NewNode.Path=[Node(CurrentNode).Path NewNode.Name];
                LastIndex=LastIndex+1;
                Node(LastIndex)=NewNode;

                Node(CurrentNode).Children=[Node(CurrentNode).Children LastIndex];
                CurrentNode=LastIndex;

            end

            Node(CurrentNode).Count=Node(CurrentNode).Count+1;

        end

    end

    % Step 3: Pattern Mining

    for i=2:numel(Node)

        % Dedicated Patterns
        S=GetPowerSet(Node(i).Path(1:end-1))';
        Node(i).Patterns=cell(size(S));         %#ok
        Node(i).PatternCount=zeros(size(S));    %#ok
        for j=1:numel(Node(i).Patterns)
            Node(i).Patterns{j}=[S{j} Node(i).Name];
            Node(i).PatternCount(j)=Node(i).Count;
        end

        % Trasfer Dedicated Patterns to Parents
        k=i;
        while true

            p=Node(k).Parent;
            if p==0
                break;
            end

            for j=1:numel(Node(i).Patterns)
                Pj=Node(i).Patterns{j};

                PatternFound=false;
                for l=1:numel(Node(p).Patterns)
                    Pl=Node(p).Patterns{l};
                    if IsSame(Pj,Pl)
                        PatternFound=true;
                        break;
                    end
                end

                if ~PatternFound
                    l=numel(Node(p).Patterns)+1;
                    Node(p).Patterns{l}=Pj;
                    Node(p).PatternCount(l)=0;
                end

                Node(p).PatternCount(l)=Node(p).PatternCount(l)+Node(i).PatternCount(j);

            end

            k=p;

        end

    end

    Patterns=Node(1).Patterns;
    PatternCount=Node(1).PatternCount;

    Patterns=Patterns(PatternCount/numel(T)>=MST);
    PatternCount=PatternCount(PatternCount/numel(T)>=MST);

    for j=1:size(Patterns,2)
        Patterns{2,j}=PatternCount(j);
    end

    % Step 4: Extract Rules

    Rules=cell(0,5);
    Supp=[];
    Conf=[];
    Lift=[];
    r=0;
    for j=1:size(Patterns,2)
        P=Patterns{1,j};
        if numel(P)<2
            continue;
        end

        countP=PatternCount(j);

        S=GetNonTrivialSubsets(P);
        Q=S(end:-1:1);
        % S{k} --> Q{k}
        for k=1:numel(S)
            for l=1:size(Patterns,2)
                if IsSame(S{k},Patterns{1,l})
                    countS=PatternCount(l);
                    break;
                end
            end
            for l=1:size(Patterns,2)
                if IsSame(Q{k},Patterns{1,l})
                    countQ=PatternCount(l);
                    break;
                end
            end

            supp=countP/numel(T);
            conf=countP/countS;
            lift=countP/(countS*countQ/numel(T));

            r=r+1;
            Rules{r,1}=S{k};
            Rules{r,2}=Q{k};
            Rules{r,3}=supp;
            Rules{r,4}=conf;
            Rules{r,5}=lift;

            Supp(r)=supp;   %#ok
            Conf(r)=conf;   %#ok
            Lift(r)=lift;   %#ok

        end

    end

    FinalRules=Rules(Conf>=MCT & Lift>=1,:);

    out.Node=Node;
    out.Patterns=Patterns;
    out.PatternCount=PatternCount;
    out.Rules=Rules;
    out.Supp=Supp;
    out.Conf=Conf;
    out.Lift=Lift;
    out.FinalRules=FinalRules;
    
end