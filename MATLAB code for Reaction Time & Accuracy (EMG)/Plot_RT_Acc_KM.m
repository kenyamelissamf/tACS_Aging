%Plotting Reaction Time and Accuracy
close all


%% List of subject IDs, session number and task
%subjectID = {'AD0109', 'RP0129', 'KM0404', 'FB0901', 'CD1107', 'PB0526', 'MB0522', 'FB0210', 'DF0720', 'PT0728', 'CB0724', 'JM0410', 'GM0804', 'TN0118', 'ME0212'};
Sessions = {'2'};
%MVCList = [55.6,50.6,40.7,35.6,23,38.3,17.6,26,39.3,31,29,41,21.3,18.3,30];

MVC = 41 * 0.15;
SubjectName = 'PT0728';
%MVC = 29;
dur = 5; % 5s task
MyPath = ['C:\Users\user\OneDrive - McGill University\Documents\Kenya_McGill\Kenya_tACS_EEG_DATA\' SubjectName '\S' Sessions{1} '_BIOPAC'];
a = dir([MyPath '/*.mat']);
for iFile = 1:length(a)
    FileName = a(iFile).name;
    cd (['C:\Users\user\OneDrive - McGill University\Documents\Kenya_McGill\Kenya_tACS_EEG_DATA\' SubjectName '\S' Sessions{1} '_BIOPAC']);
    load(FileName,'channels');
    load(FileName, 'event_markers');
    load(FileName, 'samples_per_second');


    %Create event step signal
    samples = size(channels{1,1}.data);
    eventplot = zeros(1,samples(2));
    totalevents = size(event_markers);
    start_task = 1;
    end_task = 1;

    for e = 1:totalevents(2)-1
        if strcmp(event_markers{1,e}.label, 'Begin task') == 1
            start_task = event_markers{1,e}.sample_index;
        end
        if strcmp(event_markers{1,e+1}.label, 'Show cross_b') == 1
            end_task = event_markers{1,e+1}.sample_index;
        end

        eventplot(start_task:end_task) = 1;
    end

    %% Align grip with timing
    chan = size(channels);
    data_ds = [];
    fs = 250;

    %Downsample to 250 Hz
    for m = 1:chan(2)
        data_ds(:,m) = downsample(channels{1,m}.data,10);
    end

    data_ds(:,m+1) = downsample(eventplot,10);
    time = 0:1/fs:(length(data_ds))/fs-1/fs;
    
    gripper.raw = data_ds(:,1);
    event.data = data_ds(:,11);
    emg1.raw = data_ds(:,6);
    emg2.raw = data_ds(:,9);
    event.timing = find(diff(event.data));
    event.timing = event.timing(2:length( event.timing));
    
    %Low pass filter
%     emg1.filter = lowpass(emg1.raw, 0.00005,fs);
%     plot(emg1.filter)
%     emg2.filter = lowpass(emg2.raw, 0.00005,fs);
%     plot(emg2.filter)
    
%     %Envelope
%     [up,lo] = envelope(emg1.raw,20,'peak');
%     emg1.env = up;
%     [up,lo] = envelope(emg2.raw,20,'peak');
%     emg2.env = up;

    index_start = 340*fs+1;
    index_end = 390*fs;

    figure('NumberTitle', 'off', 'Name', FileName);
    subplot 411
    plot(time(index_start:index_end),data_ds(index_start:index_end,1),'linewidth',1.1);
    title('Gripper','fontsize',14);
    subplot 412
    plot (time(index_start:index_end),data_ds(index_start:index_end,m+1),'linewidth',1.1);
    title('Event','fontsize',14);
    subplot 413
    plot (time(index_start:index_end),data_ds(index_start:index_end,6),'linewidth',1.1);
    title('EMG I','fontsize',14);
    subplot 414
    plot (time(index_start:index_end),data_ds(index_start:index_end,9),'linewidth',1.1);
    title('EMG II','fontsize',14);

    
    fprintf(strcat(FileName,' \n' ));
    for i = 1:((length(event.timing))-1)/2
        k = 2*i - 1;
        % ---- EMG1 ---- %
        gripper.pre{i} = gripper.raw(event.timing(k)-(2*fs)+1:event.timing(k));
        index = [];
        for n = 1:dur*fs      % n: time points     
            if gripper.raw(event.timing(k)+n) > (MVC/2)
                index = [index n];         % all time points when Gripper > Gripper-pre    
            end
        end
        %Adds NaN value if the EMG value never reached more thatn 2 SD
        if isempty(index) == 1
            RT{iFile}(i,1) = NaN;
        else
            RT{iFile}(i,1) = index(1)/fs;
        end
    
     end  
     %Removes NaN values
     RT{iFile} = rmmissing(RT{iFile});
    
     %% Accuracy
     if isequal(SubjectName,'FB0901') && isequal(Sessions, '2')
         MVC = 35.6 * 0.15;
     end
     for i = 1:length(event.timing)/2
         k = 2*i - 1;
         gripper.val{i} = gripper.raw(event.timing(k)+1:event.timing(k+1));
     end
     for i = 1:length(event.timing)/2
        %     ACC(i) = sqrt(sum((grip.val{i}-MVC).^2)/(length(event.timing)/2))/MVC;
        error{iFile}(i,:) = sum(sqrt((gripper.val{i}-MVC).^2)/MVC)/length(gripper.val{i});
     end

     Error{iFile} = error{iFile};
        
   
    
end

