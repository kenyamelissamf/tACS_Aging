%List of subject IDs, session number and task
%subjectID = {'AD0109', 'CB0724', 'CD1107', 'DF0720', 'FB0210', 'FB0901', 'GM0804', 'JM0410', 'KM0404', 'MB0522', 'ME0212', 'PB0526',  'PT0728', 'RP0129', 'TN0118'};
subjectID = {'Group'};
subjectIDYoung =  {'GroupYoung'};
Session = {'20', '70', 'Sham'};
Timee = {'Baseline' '15min' '45min'};
Channels = {'FC5' 'FC3' 'FC1' 'C5' 'C3' 'C1' 'CP5' 'CP3' 'CP1'};
Group_String = {' @ Group_analysis/@intra/timefreq_morlet_230605_1700_ersd_tfbands.mat',...
                ' @ Group_analysis/@intra/timefreq_morlet_230606_1547_ersd_tfbands.mat',...
                ' @ Group_analysis/@intra/timefreq_morlet_230606_1616_ersd_tfbands.mat'};

sub = size(subjectID);
ses = size(Session);
time = size(Timee);
fig = 1;
chan = size(Channels);
gs= 1;

for subject = 1:sub(2)
    for session =  1:ses(2)
        Baseline = load(strcat('PlotMRBD_',char(subjectID(subject)),'_',char(Session(session)),'_Baseline'));    
        Fifteen = load(strcat('PlotMRBD_',char(subjectID(subject)),'_',char(Session(session)),'_15min'));
        Fortyfive = load(strcat('PlotMRBD_',char(subjectID(subject)),'_',char(Session(session)),'_45min'));
        basevar = genvarname(strcat('PlotMRBD_',char(subjectID(subject)),'_',char(Session(session)),'_Baseline'));
        fifvar = genvarname(strcat('PlotMRBD_',char(subjectID(subject)),'_',char(Session(session)),'_15min'));
        forvar = genvarname(strcat('PlotMRBD_',char(subjectID(subject)),'_',char(Session(session)),'_45min'));

%             channels = Baseline.RowNames;
%             chan = size(channels);
            figure('NumberTitle', 'off', 'Name', strcat(char(subjectID(subject)), ' ', char(Session(session))));
            for channel=1:chan(2)
                if strcmp(subjectID(subject), 'Group') == 1
                    if ismember(strcat(Channels(channel),Group_String(gs)), Baseline.RowNames) == 1
                        index = find(strcmp(Baseline.RowNames, strcat(Channels(channel),Group_String(gs))));
                        subplot(3,3,channel)
                        plot(Baseline.Time, Baseline.TF(index,:),Fifteen.Time,Fifteen.TF(index,:),Fortyfive.Time, Fortyfive.TF(index,:), 'LineWidth',2);
                        title(string(Channels(channel)))
                        xlabel('Time(s)')
                        ylabel('MRBD(%)')
                    else
                        subplot(3,3,channel)
                    end
                else
                    if ismember(Channels (channel), Baseline.RowNames) == 1
                        index = find(strcmp(Baseline.RowNames, Channels(channel)));
                        subplot(3,3,channel)
                        plot(Baseline.Time, Baseline.TF(index,:),Fifteen.Time,Fifteen.TF(index,:),Fortyfive.Time, Fortyfive.TF(index,:), 'LineWidth',2);
                        title(string(Channels(channel)))
                        xlabel('Time(s)')
                        ylabel('MRBD(%)')
                    else
                        subplot(3,3,channel)
                    end
                end
            end
            
            legend('Baseline', '15min', '45min')
            fig = fig + 1;
            if strcmp(subjectID(subject), 'Group') == 1
                gs = gs +1;
            end
        end
end


%Plot Young VS Old Baseline First Session
Young = load('PlotMRBD_GroupYoung_Baseline');    
Old = load('PlotMRBD_GroupOld_Baseline');

GroupYVO_String = {' @ Group_analysis/@intra/timefreq_morlet_230627_2311_ersd_tfbands.mat'};
TimeYoung = (Young.Time-1);
gf = 1;
figure('NumberTitle', 'off', 'Name','YoungVSOld');
for channel=1:chan(2)
    if ismember(strcat(Channels(channel),GroupYVO_String(gf)), Young.RowNames) == 1
        index = find(strcmp(Young.RowNames, strcat(Channels(channel),GroupYVO_String(gf))));
        subplot(3,3,channel)
        plot(TimeYoung, Young.TF(index,:),Old.Time,Old.TF(index,:), 'LineWidth',2);
        colororder(["#7E2F8E"; "#4DBEEE"])
        title(string(Channels(channel)))
        xlabel('Time(s)')
        ylabel('MRBD(%)')
    else
        subplot(3,3,channel)
    end
end

legend('Young', 'Old')
fig = fig + 1;
if strcmp(subjectID(subject), 'Group') == 1
    gf = gf +1;
end












