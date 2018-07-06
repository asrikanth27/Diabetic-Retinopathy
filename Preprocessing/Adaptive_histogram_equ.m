clear;
clc;
dir1='D:\test_dat\test'
dinfo = dir(dir1);
names_cell = {dinfo.name};
iter=1;
for n1=3:length(names_cell)
   
     filen=names_cell(n1);
     file=char(filen);
     im=imread(file);
     im=imresize(im,[224,224]);
     im=im2double(im);
     
     
     im(:,:,2)=adapthisteq(im(:,:,2));
     s3='D:\test_dat\test_processed\'
     s4=strcat(s3,file);
     imwrite(im,s4)
     iter
     iter=iter+1;
      
     
end
     

    
    
