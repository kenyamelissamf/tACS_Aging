% Abr 06, 2023, Kenya M.
% EMG1 integrate: 7
% EMG2 intergrate:10

clear
clc
close all

%% List of subject IDs, session number and task
subjectID = {'AD0109', 'CB0724', 'CD1107', 'DF0720', 'FB0210', 'FB0901', 'GM0804', 'JM0410', 'KM0404', 'MB0522', 'ME0212', 'PB0526',  'PT0728', 'RP0129', 'TN0118'};
Sessions = {'20', '70', 'Sham'};
Timee = {'Baseline' 'tACS' '15min' '45min'};

%% load data
SubjectName = 'AD0109';
MVC = 20 * 0.15;
Session = 2;
dur = 5; % 5s task
MyPath = ['C:\Users\user\OneDrive - McGill University\Documents\Kenya_McGill\Kenya_tACS_EEG_DATA\' SubjectName '\S' num2str(Session) '_BIOPAC\'];
a = dir([MyPath '/*.mat']);


for iFile = 1:length(a)
    FileName = a(iFile).name;
    cd (['C:\Users\user\OneDrive - McGill University\Documents\Kenya_McGill\Kenya_tACS_EEG_DATA\' SubjectName '\S' num2str(Session) '_BIOPAC\']);
    load (FileName,'data');


    %% Align grip with timing
    data_ds = downsample(data,10);
    fs = 250;
    
%     figure;
%     for i = 1:11
%         subplot (11,1,i);
%         plot(data_ds(50*fs+1:100*fs,i));
%     end
    
    figure;
    subplot 411
    plot(data_ds(50*fs+1:100*fs,1),'linewidth',1.1);
    title('Gripper','fontsize',14);
    subplot 412
    plot (data_ds(50*fs+1:100*fs,2),'linewidth',1.1);
    title('Event','fontsize',14);
    subplot 413
    plot (data_ds(50*fs+1:100*fs,7),'linewidth',1.1);
    title('EMG I','fontsize',14);
    subplot 414
    plot (data_ds(50*fs+1:100*fs,10),'linewidth',1.1);
    title('EMG II','fontsize',14);
    
    %% Accuracy
    gripper.raw = data_ds(:,1);
    event.data = data_ds(:,2);
    emg1.raw = data_ds(:,7);
    emg2.raw = data_ds(:,10);
    event.timing = find(diff(event.data));
    for i = 1:length(event.timing)/2
        k = 2*i - 1;
        gripper.val{i} = gripper.raw(event.timing(k)+1:event.timing(k+1));
    end
    for i = 1:length(event.timing)/2
        %     ACC(i) = sqrt(sum((grip.val{i}-MVC).^2)/(length(event.timing)/2))/MVC;
        error{iFile}(i,:) = sum(sqrt((gripper.val{i}-MVC).^2)/MVC)/length(gripper.val{i});
    end
    
    
    %% Reaction time
    for i = 1:length(event.timing)/2
        k = 2*i - 1;
        % ---- EMG1 ---- %
        emg1.pre{i} = emg1.raw(event.timing(k)-fs+1:event.timing(k));
        emg2.pre{i} = emg2.raw(event.timing(k)-fs+1:event.timing(k));
        index = [];
        for n = 1:dur*fs      % n: time points     
            if emg1.raw(event.timing(k)+n) > mean(emg1.pre{i})+2*std(emg1.pre{i}) && ...
                    emg2.raw(event.timing(k)+n) > mean(emg2.pre{i})+2*std(emg2.pre{i})
                index = [index n];         % all time points when EMG > EMG-pre    
            end
        end
        RT{iFile}(i,1) = index(1)/fs;
    end
    
end
[h.etacs,p.etacs]=ttest2(error{3},error{4});
[h.e15min,p.e15min]=ttest2(error{3},error{1});
[h.e45min,p.e45min]=ttest2(error{3},error{2});

[h.rtacs,p.rtacs]=ttest2(RT{3},RT{4});
[h.r15min,p.r15min]=ttest2(RT{3},RT{1});
[h.r45min,p.r45min]=ttest2(RT{3},RT{2});

