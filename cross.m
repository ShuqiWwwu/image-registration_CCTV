
rootdir='/Users/shuqiwu/Documents/MATLAB/111/';
subdir=dir(strcat(rootdir,'*.jpg'));
I = cell(1,length(subdir));
data = repmat(uint8(0), 266, 348, length(subdir));
aa = 0;
I{1} = imread(strcat(rootdir,'1312374193.jpg'));
I{1} = imcrop(I{1},[5 23 348 288]);
data(:,:,1) = rgb2gray(I{1}); 
for i = 2:length(subdir)
    ImageName = subdir(i).name;
    I{i} = imread(strcat(rootdir,ImageName));
    I{i} = imcrop(I{i},[5 23 348 288]);
    data(:,:,i) = rgb2gray(I{i}); 
    

end

template = data(:,:,1); 
template = imgaussfilt(template)
template = edge(template,'canny');
template = template - mean(mean(template));

for ii = 2:length(subdir)
    imggauss = imgaussfilt(data(:,:,ii))
    imgedge = edge(imggauss,'canny');
    img = imgedge - mean(mean(imgedge));
    crr = xcorr2(img, template);
    [y,x] = find(crr == max(crr(:)));
    [ssr,snd] = max(abs(crr(:)));
    [ypeak, xpeak] = ind2sub(size(crr),snd(1));
    corr_offset = [(ypeak - size(template,1)) (xpeak - size(template,2))];
    %ab_value = sqrt(corr_offset(1)^2+corr_offset(2)^2);
    offset1 = sqrt(corr_offset(1)^2);
    offset2 = sqrt(corr_offset(2)^2);
    if (offset1<90) && (offset2<90)
    %if (ab_value < 20)
        imgnew = imtranslate(I{ii},[-corr_offset(2),-corr_offset(1)]); % image translate
        saveddir='/Users/shuqiwu/Documents/MATLAB/112';
        savedname=fullfile(saveddir,subdir(ii).name);
        imwrite(0.5*imgnew + 0.5*I{1},savedname);
      
    else 
        saveddir='/Users/shuqiwu/Documents/MATLAB/113';
        savedname=fullfile(saveddir,subdir(ii).name);
        imwrite(I{ii},savedname);
        
    end
    
    %subplot (1,2,1), imshow (I{1}/2+I{2}/2);
    %subplot (1,2,2), imshow (0.5*imgnew + 0.5*I{1});

    %subplot (2,2,3), imshow (I{1}/2+I{2}/2);
    %subplot (2,2,4), imshow (I{1}/2+imgnew/2);
    
    
    
end


