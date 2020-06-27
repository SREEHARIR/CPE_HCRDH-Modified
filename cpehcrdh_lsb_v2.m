clc;
close all;
clear;

[file,path]=uigetfile('*.pgm','Select image file');
ss=strcat(path,file);
img=imread(ss);

%% Image Standardisation %%
if size(img,3) == 3
    I = rgb2gray(img);
else
    I = img;
end
I = imresize(I,[512 512]);
I = double(I);
figure('Name','Original Image','NumberTitle','off');
imshow(uint8(I));

%% Preprocess Prediction Error %%

Iinv = bitset(I,8,~bitget(I,8));
% figure('Name','I inverse','NumberTitle','off');
% imshow(uint8(Iinv));

error = zeros(512);
pred = zeros(512);
In=zeros(512);
for ii = 1:512
    for jj = 1:512
        if ii == 1 || jj == 1
            pred(ii,jj) = I(ii,jj);
        else
            pred(ii,jj) = floor((pred(ii-1,jj) + pred(ii,jj-1))/2);
        end
        delta = abs(pred(ii,jj) - I(ii,jj));
        deltainv = abs(pred(ii,jj) - Iinv(ii,jj));
%         if(delta > deltainv)
% %             if(I(ii,jj)<128)
% %                 In(ii,jj) = pred(ii,jj) - 62; % used to be - 63
% %             else
% %                 In(ii,jj) = pred(ii,jj) + 62;
% %             end
%             In(ii,jj) = I(ii,jj) + ((6+delta-deltainv)*(((I(ii,jj)<Iinv(ii,jj))*2)-1));
%             error(ii,jj) = 1;
%         elseif delta == deltainv
%             In(ii,jj) = I(ii,jj) + ((6+delta-deltainv)*(((I(ii,jj)<Iinv(ii,jj))*2)-1));
%             error(ii,jj) = 1;
%         elseif delta - deltainv >= -4
        if delta - deltainv >= -4
            In(ii,jj) = I(ii,jj) + ((6+delta-deltainv)*(((I(ii,jj)<Iinv(ii,jj))*2)-1));
            error(ii,jj) = 1;
        else
            In(ii,jj) = I(ii,jj);
        end
        pred(ii,jj) = In(ii,jj);
    end
end
% figure('Name','Prediction Corrected vs Original','NumberTitle','off');
% imshow(In~=I,[]);

%% Encryption %%

Ke = inputdlg({'Enter the Key for Encrypting Image'},'Image Encryption Key',[1,35],{'123456789'});
Ke = str2double((cell2mat(Ke)));
seed = Ke;
rng(seed,'twister');
S = randi(255,512);

Ie = bitxor(S,In);
figure('Name','Encrypted Image','NumberTitle','off');
imshow(uint8(Ie));

%% Data Embedding %%

data = inputdlg({'Enter the Data to be Embedded'},'Embedding Data',[5,35],{'This is the data to be embedded'});
data = (cell2mat(data));
data = [data '..........'];

M = [data, (zeros(1,floor(2*512*512/8) - numel(data)))];

Kw = inputdlg({'Enter the Key for Encrypting Word'},'Word Encryption Key',[1,35],{'123456'});
Kw = str2double((cell2mat(Kw)));
seed = Kw;
rng(seed,'twister');
S = randi(255,[1,numel(M)]);

M = double(M);
Me = bitxor(M,S);
Me = Me';
Me = dec2bin(Me);
Me = Me';
Me = reshape(Me,[1,numel(Me)]);
Me = double(Me);
Me = Me - 48;

Iew = Ie;
m=0;
for ii = 2:512
    for jj = 2:512
        Iew(ii,jj) = bitset(Iew(ii,jj),8,Me(m+1));
        m = m + 1;
    end
end
for ii = 1:512
    for jj = 1:512
        Iew(ii,jj) = bitset(Iew(ii,jj),1,Me(m+1));
        m = m + 1;
    end
end

figure('Name','Encrypted Image with hidden word','NumberTitle','off');
imshow(uint8(Iew));
%% Differential Encoding

I_diff = zeros(512);
for i=1:512
    for j = 1:512
        if j==1
            I_diff(i,j) = Iew(i,j);
            continue;
        end
        I_diff(i,j) = Iew(i,j) - Iew(i,j-1);
    end
end

figure;
imshow(I_diff+255,[]);
figure;
histogram(Iew,256);
figure;
histogram(I_diff,256);


%% Compression/Encoding

img_to_compress = I_diff + 255;

% bins = zeros(1,256);
bins = zeros(1,512);
for i = 1:size(img_to_compress,1)
    for j = 1:size(img_to_compress,2)
        val = img_to_compress(i,j)+1;
        bins(val) = bins(val) + 1;
    end
end

% symbols = 0:255;
symbols = 0:511;
prob = bins / (size(img_to_compress,1)*size(img_to_compress,2));

set(0,'RecursionLimit',600);
dict = huffmandict(symbols,prob);

fprintf('Encoding Image\n\n');

% img_to_compress_signal = reshape(img_to_compress,[1,size(img_to_compress,1)*size(img_to_compress,2)]);
img_signal = img_to_compress(:);
img_encoded = huffmanenco(img_signal,dict);

% first 50 bits = length of image size1 
% 51:100 = breadth of image size2
% 101:32996 = dictionary saved for 1:256

header = [];
header = [header dec2bin(size(img_to_compress,1),50)];
header = [header dec2bin(size(img_to_compress,2),50)];
% header = [header char(reshape(cell2mat(dict(:,2)),[1,2048])+48)];

img_encoded_with_header = [header (img_encoded'+48)];

fprintf('Size before Compression: %i bits\n',numel(img_to_compress)*8);
fprintf('Size after Compression: %i bits\n',numel(img_encoded_with_header));
fprintf('Compression ratio: %2.6f\n\n',numel(img_to_compress)*8/numel(img_encoded_with_header));

%% Extraction %%
%% Uncompressing/Decoding

% header = img_encoded_with_header(1:2148);
header = img_encoded_with_header(1:100);
% img_encoded = double(img_encoded_with_header(2149:end)-48);
img_encoded = double(img_encoded_with_header(101:end)-48);

size1 = bin2dec(header(1:50));
size2 = bin2dec(header(51:100));
% dict = mat2cell([double(0:255)' double(reshape((header(101:end)-48),[256,8]))],ones(1,256),[1,8]);

fprintf('Decoding Image\n');
img_decoded_signal = huffmandeco(img_encoded,dict);
fprintf('Decoding Finished\n');
img_decoded = reshape(img_decoded_signal,[size1,size2]);

if ~isequal(img_to_compress,img_decoded)
    fprintf('Not Lossless Compression\n');
end

% Iew = img_decoded;

%% Differential Decoding

I_diff = img_decoded - 255;
% I_diff = zeros(512);
for i=1:512
    for j = 1:512
        if j==1
            Iew(i,j) = I_diff(i,j);
            continue;
        end
        Iew(i,j) = I_diff(i,j) + Iew(i,j-1);
    end
end


%% Message Extraction %%

m = 0;
Me = zeros([1 512*512]);
for ii = 2:512
    for jj = 2:512
        Me(m+1) = bitget(Iew(ii,jj),8);
        m = m + 1;
    end
end
for ii = 1:512
    for jj = 1:512
        Me(m+1) = bitget(Iew(ii,jj),1);
        m = m + 1;
    end
end
Me = Me(1:261120); %% floor((512*512-512-511)/8)*8

Kw = inputdlg({'Enter the Key for Decrypting Word'},'Word Decryption Key',[1,35],{'123456'});
Kw = str2double((cell2mat(Kw)));
seed = Kw;
rng(seed,'twister');
S = randi(255,[1,numel(Me)/8]);

Me = Me + 48;
Me = char(Me);
Me = reshape(Me,[8,numel(Me)/8]);
Me = Me';
Me = bin2dec(Me);
Me = (Me');
Md = bitxor(Me,S);
Md = char(Md);

Md = strsplit(Md,'...');
Md = cell2mat(Md(1));
fprintf(Md);
fprintf('\n\n');
msgbox(Md,'Decoded Message');

%% Image Extraction %%

Ke = inputdlg({'Enter the Key for Decrypting Image'},'Image Decryption Key',[1,35],{'123456789'});
Ke = str2double((cell2mat(Ke)));
seed = Ke;
rng(seed,'twister');
S = randi(255,512);
Id = bitxor(S,Iew);
% figure('Name','Decoded Image','NumberTitle','off');
% imshow(uint8(Id));

%figure('Name','Correcting the Decoded Image','NumberTitle','off');
for ii = 2:512
    for jj  = 2:512
        predictor = floor((Id(ii-1,jj) + Id(ii,jj-1))/2);
        Id0m = bitset(Id(ii,jj),8,0);
        Id1m = bitset(Id(ii,jj),8,1);
        delta0 = abs(predictor - Id0m);
        delta1 = abs(predictor - Id1m);
        if delta0 < delta1
            Id(ii,jj) = bitset(Id(ii,jj),8,0);
        elseif delta0 == delta1
%             fprintf('equal delta decode [%i %i]\n',ii,jj);
            Id(ii,jj) = bitset(Id(ii,jj),8,0);
        else
            Id(ii,jj) = bitset(Id(ii,jj),8,1);
        end
    end
%     imshow(uint8(Id));
end


% figure('Name','Predicted','NumberTitle','off');
% imshow(uint8(pred));

figure('Name','Corrected Decoded Image','NumberTitle','off');
imshow(uint8(Id));

Id = bitset(Id,1,bitget(I,1));
figure('Name','Id vs I Uncorrected','NumberTitle','off')
imshow((Id~=I),[]);
% imshow((Id~=In),[]);

%% PSNR and SSIM

img_processed = Id;
img_original = I;
psnrval = psnr(img_processed,img_original,255);
ssimval = ssim(img_processed,img_original);
fprintf('psnr = %2.2f dB \nssim = %1.5f \n',psnrval,ssimval);
% msgbox({['PSNR =  ',num2str(psnrval),' dB'],['SSIM =  ',num2str(ssimval)]},'PSNR and SSIM');
bpp = (2*512*512 - 512 - 512 + 1)/(512*512);
fprintf('Max bits per pixel = %1.6f \n',bpp);