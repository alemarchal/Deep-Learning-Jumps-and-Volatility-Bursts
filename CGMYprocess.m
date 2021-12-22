% Produces CGMY paths , us ing a sub o r d ina t o r r e p r e s e n t a t i o n
% based on [32] and [36]
% OUTPUTS:
% Xout = the p roc e s s v a l u e s
% t = a cor r e sponding f i x e d i n t e r v a l t ime g r i d
% INPUTS:
% C,G,M,Y = the p roc e s s parameters
% n = the number o f paths
% p = the number of t ime s t e p s
% T = the f i n a l t ime
 % t0 = the i n i t i a l t ime


 
 function [Xout]=CGMYprocess(C,G,M,Y, n , p ,T, t0 )

 if nargin <8
 t0=0;
 end

 A=(G-M) / 2 ;
 B=(G+M) / 2 ;

 epsilon=1E-4; %The jump t runc a t i on l e v e l
 K=C*2^(-Y/2)*sqrt(pi)/gamma(Y/2+0.5) ; %from Equat ion (4.14)

d=K* epsilon^(1-Y/2)/(1-Y/2) ; %Equat ion (4.21)
lambda=2*K*epsilon^(-Y/2)/Y ; %Equat ion (4.20)

 %Pre allocate memory :
Xout=zeros(n, p) ;
for i =1:n
 %The jump times :
 tj=t0 ;
 while tj(end)<T
   U2 = rand(1 ,2*round( lambda*(T-t0 ) ) ) ;
   int = -log (U2)/lambda ;
   tj = [tj tj(end)+cumsum(int)] ;
 end
 tj = tj(tj<T) ;
 
 
 %Applying (4.19) f o r the jump s i z e s :
 U1 = rand(1, length(tj)-1);
 yj =[0 ,epsilon./(U1).^(2/Y)] ;
 
 
 %Applying the r e j e c t i o n (Theorem 4.7 ) :
 U3 = rand(size(yj));
 Zt = d*tj + cumsum(yj.*(f(yj)>U3));
 
 
 % Performing the sub o r d ina t i on :
 dZ = Zt(2:end)-Zt(1:end-1);
 W = [0 cumsum(sqrt(dZ).*randn(size(dZ)))] ;
 X = A*Zt+W;


%A MEX-C file, confining the process to a fixed time grid :
%[Xout(i,:),t] = GridTime(X, tj, p, T) ;
%clc
%disp ( [ 'Path Generation ' num2str(100*i/n) '% Complete '])

 [Xout(i,:)] = GridTime(X, tj, p, T) ;


end


% Equat ion (4.13)
function out=f(t)
    out=2^(Y/2)*gamma(Y/2+0.5)*exp(A^2*t/2-B^2*t/4) ...
    .*D(-Y,B*sqrt(t))/sqrt(pi);
end




function [Xout] = GridTime(X, tj, p, T) %GridTime is our function that we use to squeeze or stretch time
   
    
    tj_calend = floor(tj*p/T);
     Xout(1) = X(1);
        
     if p > length(X) %if we want more points than the there are jumps in the subordinator
        for j = 2:p
            if ismember(j, tj_calend)
                Xout(j) = X(max(find(tj_calend==j)));
            else
                Xout(j)=Xout(j-1);
            end
        end
        
        
     elseif    p < length(X)
        
         disp('case p < length(X)')
         
         for j = 1:p
             Xout(j) = X(ceil(j*length(X)/p));
         end
                 
     else
         
        Xout=X;
         
        
     end
     
     
     
end


Xout=Xout';

 end
 
 
 %Implementat ion of f ( x ) in term o f c o n f l u e n t hype r g eome t r i c f u n c t i o n s :
 function out=D(nu , z )
 %A parabolic cylinder function :
 out=2^(nu/2).*U(-nu/2 , 0.5 , z.^2 /2).*exp(-z.^2/4 ) ;
 end

 function out=U( a , b , z )
 out=zeros(size(z)) ;
 %A c o n f l u e n t hype r g eome t r i c f unc t i on of the second kind :
 cut=10;
 out(z<cut)=(pi/sin(pi*b)) * (HYPERGEOM(a , b , z(z<cut))/ ...
 (gamma(a-b+1)*gamma(b))-z(z<cut ).^(1-b) ...
 .*HYPERGEOM(a-b+1,2-b , z(z<cut))/(gamma(a)*gamma(2-b) ) ) ;
 %An a l t e r n a t e implementat ion wi th b e t t e r convergence
 %f o r l a r g e v a l u e s o f z :
 Z=z(z>=cut) ;
 temp=zeros(size(Z)) ;
 for i=1:length(Z)
 fun= @(t) exp(-Z(i)*t).*t.^(a-1).*(1+t).^(b-a-1);
 temp(i)= quad(fun, 0 , 1E6 , 1E-4)/gamma(a) ;
 end

 out(z>=cut)=temp ;
 end

 function out=HYPERGEOM(a, b , z , tol)
 %A c o n f l u e n t hype r g eome t r i c f unc t i on of the f i r s t kind :
 if nargin<4
 tol=1E-3;
 end

 out=1;
 term=ones(size(z)) ;

 n=1;
 while max(term)>tol || n<100
 term=term.*((a+n-1)*z/(n*(b+n-1) ) ) ;
 out=out+term ;
 n=n+1;
 end

 end