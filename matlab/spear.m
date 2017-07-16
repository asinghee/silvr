function [r,t,p]=spear(x,y)
%Syntax: [r,t,p]=spear(x,y)
%__________________________
%
% Spearman's rank correalation coefficient.
%
% r is the Spearman's rank correlation coefficient.
% t is the t-ratio of r.
% p is the corresponding p-value.
% x is the first data series (column).
% y is the second data series (column), which can contain multiple columns.
%
%
% Reference:
% Press W. H., Teukolsky S. A., Vetterling W. T., Flannery B. P.(1996):
% Numerical Recipes in C, Cambridge University Press. Page 641.
%
%
% Alexandros Leontitsis
% Department of Education
% University of Ioannina
% 45110- Dourouti
% Ioannina
% Greece
%
% University e-mail: me00743@cc.uoi.gr
% Lifetime e-mail: leoaleq@yahoo.com
% Homepage: http://www.geocities.com/CapeCanaveral/Lab/1421
%
% 3 Feb 2002.


% x and y myst have equal number of rows
if size(x,1)~=size(y,1)
    error('x and y must have equal number of rows.');
end


% Get the ranks of x
R=crank(x)';

% Remove their mean
R=R-mean(R);

for i=1:size(y,2)

    % Get the ranks of y
    S=crank(y(:,i))';

    % Remove their mean
    S=S-mean(S);

    % Calculate the correlation coefficient
    r(i)=sum(R.*S)/sqrt(sum(R.^2)*sum(S.^2));

end

% Find the data length
N=length(x);

% Calculate the t statistic
t=r.*sqrt((N-2)./(1-r.^2));

% Handle the NANs
t(find(isnan(t)==1))=0;

% Calculate the p-values
p=2*(1-tcdf(abs(t),N-2));





function r=crank(x)
%Syntax: r=crank(x)
%__________________
%
% Assigns ranks on a data series x.
%
% r is the vector of the ranks
% x is the data series. It must be sorted.
%
%
% Reference:
% Press W. H., Teukolsky S. A., Vetterling W. T., Flannery B. P.(1996):
% Numerical Recipes in C, Cambridge University Press. Page 642.
%
%
% Alexandros Leontitsis
% Department of Education
% University of Ioannina
% 45110- Dourouti
% Ioannina
% Greece
%
% University e-mail: me00743@cc.uoi.gr
% Lifetime e-mail: leoaleq@yahoo.com
% Homepage: http://www.geocities.com/CapeCanaveral/Lab/1421
%
% 3 Feb 2002.

x(end+1)=max(x)+1;

for i=1:size(x,2)

    [x(:,i),z1]=sort(x(:,i));
    [z1,z2]=sort(z1);

    if var(x(:,i))==0
        r=1:size(x,1);
        return
    end

    j=1;
    while j<size(x,1)

        if x(j+1)>x(j)

            r(j)=j;

        else

            jt=0;
            while x(j+1)==x(j)

                jt=jt+1;
                j=j+1;
            end

            r1=mean(j-jt:j);

            r(j-jt:j)=r1;
        end

        j=j+1;

    end

    if j==size(x,1)

        r(size(x,1))=size(x,1);

    end


end

r=r(z2);
r(find(r==max(r)))=[];

