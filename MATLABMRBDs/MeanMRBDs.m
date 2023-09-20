%List of subject IDs, session number and task
%subjectID = {'AD0109', 'CB0724', 'CD1107', 'DF0720', 'FB0210', 'FB0901', 'GM0804', 'JM0410', 'KM0404', 'MB0522', 'ME0212', 'PB0526',  'PT0728', 'RP0129', 'TN0118'};
subjectID = {'AD0109'};
Session = {'20', '70', 'Sham'};
Timee = {'Baseline' '15min' '45min'};
Channels = {'FC3' 'FC1' 'C5' 'C3' 'C1' 'CP5' 'CP3' 'CP1'};

sub = size(subjectID);
ses = size(Session);
time = size(Timee);
fig = 1;
chan = size(Channels);

for subject = 1:sub(2)
    for session =  1:ses(2)
        Baseline = load(strcat('MeanMRBD_',char(subjectID(subject)),'_',char(Session(session)),'_Baseline'));    
        Fifteen = load(strcat('MeanMRBD_',char(subjectID(subject)),'_',char(Session(session)),'_15min'));
        Fortyfive = load(strcat('MeanMRBD_',char(subjectID(subject)),'_',char(Session(session)),'_45min'));
     
        i = 1;
        MRBD{subject}{session}{i} = Baseline.Values;
        MRBD{subject}{session}{i+1} = Fifteen.Values;
        MRBD{subject}{session}{i+3} = Fortyfive.Values;
    end
end




