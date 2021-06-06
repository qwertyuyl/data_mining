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

function PlotTree(Node)

    [x, y]=treelayout([Node.Parent]);
    
    for i=1:numel(Node)
        j=Node(i).Parent;
        if j~=0
            plot([x(i) x(j)],[y(i) y(j)],'b','LineWidth',1);
        end
        hold on;
    end
    
    for i=1:numel(Node)
        plot(x(i),y(i),'bo','MarkerFaceColor','w','MarkerSize',28,'LineWidth',1);
        text(x(i),y(i),...
            [num2str(Node(i).Name) ':' num2str(Node(i).Count)],...
            'HorizontalAlignment','center');
    end
    
    axis off;

end