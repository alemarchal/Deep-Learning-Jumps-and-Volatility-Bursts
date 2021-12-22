function v=ToColumn(p) % transforms a vector to a column vector 
[s1 s2]=size(p);
if s1==1
    p=p.';
end
v=p;
end