function createDatasetVect(cardPoints)
   local cardPoints = cardPoints or 1000

   local pos = torch.Tensor({1,1})
   local neg = torch.Tensor({-1,-1})

   local var_positive=1.0

   local X=torch.Tensor(cardPoints,2)
   local y=torch.Tensor(cardPoints,2)

   for i=1,cardPoints/2 do
      
      X[i]= pos + torch.randn(2)/1.5
      y[i] = torch.Tensor({1,0})
   end

   for i=cardPoints/2+1,cardPoints do
      X[i]= neg + torch.randn(2)/1.5
      y[i] = torch.Tensor({0,1})
   end

   return X,y
end 

function createDatasetScalar(cardPoints)
   local cardPoints = cardPoints or 1000

   local pos = torch.Tensor({1,1})
   local neg = torch.Tensor({-1,-1})

   local var_positive=1.0

   local X=torch.Tensor(cardPoints,2)
   local y=torch.Tensor(cardPoints,1)

   for i=1,cardPoints/2 do
      X[i]= pos + torch.randn(2)/1.5
      y[i] = 1

   end

   for i=cardPoints/2+1,cardPoints do
      X[i]= neg + torch.randn(2)/1.5
      y[i] = 2
   end

   return X,y
end 


function plot_points(xs,ys)  
   local nb_positive=ys:eq(1):sum()
   local nb_negative=ys:eq(2):sum()
   local xs_positive=torch.Tensor(nb_positive,2)
   local xs_negative=torch.Tensor(nb_negative,2)
   local pos_positive=1
   local pos_negative=1
   for i=1,xs:size(1) do
      if (ys[i][1]==1) then xs_positive[pos_positive]:copy(xs[i]); pos_positive=pos_positive+1 
      else xs_negative[pos_negative]:copy(xs[i]); pos_negative=pos_negative+1 end
   end
   
   return {{xs_positive:t()[1],xs_positive:t()[2],"with points"},{xs_negative:t()[1],xs_negative:t()[2],"with points"}}
end

function plot_model(model,x_min,x_max,y_min,y_max,step)
   local all_points={}
   local pos=1
   
   local p=torch.Tensor(2)
   for x=x_min,x_max,step do
      p[1]=x; p[2]=y_min; local v=model:forward(p)[1]
      if (v<0) then v=-1 else v=1 end
      for y=y_min+step,y_max,step do
         p[2]=y
         local val=model:forward(p)[1]
         if (val*v<=0) then 
            v=-v
            all_points[pos]={x,y}
            pos=pos+1
         end
      end
   end
   
   for y=y_min,y_max,step do
      p[1]=x_min; p[2]=y; local v=model:forward(p)[1]
      if (v<0) then v=-1 else v=1 end
      for x=x_min+step,x_max,step do
         p[1]=x
         local val=model:forward(p)[1]
         if (val*v<0) then 
            v=-v
            all_points[pos]={x,y}
            pos=pos+1
         end
      end
   end  
   
   local t=torch.Tensor(all_points)
   if (t:size(1)>0) then return {t:t()[1],t:t()[2],"with points"} end
end

function plot(xs,ys,model,precision)
   assert(xs:size(2)==2,"Dimension must be 2 for plotting")
   if (precision==nil) then precision=100 end
   local p1=plot_points(xs,ys,nil)
   local p2=plot_model(model,xs:t()[1]:min(),xs:t()[1]:max(),xs:t()[2]:min(),xs:t()[2]:max(),(xs:t()[1]:max()-xs:t()[1]:min())/precision)
   gnuplot.plot(p2,unpack(p1))
end


