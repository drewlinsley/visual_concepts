%% Visual Concepts: Human 2AFC Behavioral Experiment 
% Deep Learning Hackathon 2016 

%% Startup & participant information 
sca; clc; clear; 
rng('default'); rng('shuffle'); % randomize the seed for pseudorandomisation

% prompt for subject information in command window and save responses in final data struct ('results'). 
computer = input('1 = mac, 2 = windows: ','s');
results.subjectID = input('Enter Subject ID (NUM_INIT): ','s');
results.subjectAge = input('Enter Subject Age: ','s');
results.subjectGender = input('Enter Subject Gender (M/F/NB): ','s'); 
screenSize = input('Full Screen Mode (0/1): '); % 0 = small screen (for debugging), 1 = full screen (for experiment)

%% Display welcome screen while task is prepared
Screen('Preference', 'SkipSyncTests', 1)
whichscreen = 0;
smallScreen = [0 0 1000 750]; 
switch screenSize
    case 0 % opens smallScreen (debugging) 
        [window, rect] = Screen(whichscreen,'OpenWindow', [], smallScreen); 
    case 1 % opens fullScreen (experiment) 
        [window, rect] = Screen(whichscreen,'OpenWindow');  
end

[xcenter, ycenter] = RectCenter(rect); % gets coordinates of center
white = WhiteIndex(window); % white pixel intensity 
black = BlackIndex(window); % black pixel intensity 
Screen('TextSize', window, 30); 
Screen('TextFont',window,'Helvetica');

[~, ~, ~] = DrawFormattedText(window, 'Preparing the task...' , 'center', 'center', black ); 
Screen(window, 'Flip');
WaitSecs(1);

%% Specify experiment parameters 

% -- EDIT FILEPATHS ('exp_images' won't be necessary later) ------------- % 
% negpath = '...negative_examples/exp_images';
% pospath = '...positive_examples/exp_images';
% instructpath = '...instructions.jpg';
negpath = '/Users/Celia/Documents/Brown Semester 7/Deep Learning/visual_reasoning/negative_examples/exp_images';
pospath = '/Users/Celia/Documents/Brown Semester 7/Deep Learning/visual_reasoning/positive_examples/exp_images';
instructpath = '/Users/Celia/Documents/Brown Semester 7/Deep Learning/visual_reasoning/instructions.jpg';
% ----------------------------------------------------------------------- % 

% -- ADJUST AS NECESSARY  ----------------------------------------------- % 
% numBlocks = # concepts, numTrials = # images per concept
% NOTE: a 'block' is a concept task (ex. 'is the dog within, y/n?')
numBlocks = 2; 
numTrials = 48; 
% ----------------------------------------------------------------------- % 

% randomly select indices of concept folders to use 
folderInds = randi([3 68],1,numBlocks);

% -- ADJUST AS NECESSARY ------------------------------------------------ %
% NOTE: respWindow & stimDuration don't have to be equal 
respWindow = 0.5; % seconds subject is given to respond
stimDuration = 0.5; % seconds image is on screen
fixDuration = 1; % seconds feedback/fixation cross is on screen 
% ----------------------------------------------------------------------- % 

if computer == '1' % mac keycodes 
    respKeys = [54 55]; % 54 = < (yes), 55 = > (no) 
    spacebar = 44;
    escape = 41;
else % windows keycodes 
    respKeys = [188 190]; % 188 = < (yes), 190 = > (no) 
    spacebar = 32;
    escape = 27;
end

textSize = 30;
fontColor = black;
yesLocation = [xcenter-400,ycenter];
noLocation = [xcenter+400,ycenter];

% add image folders to path and save a list of their contents 
addpath(negpath,pospath);
negFolder = dir(negpath);
posFolder = dir(pospath);

% save concept folder names [to be used on current subject] in 'allFolders'
for file = 3:68
    % skip empty folders (in case files downloaded incorrectly)
    if isempty(dir(fullfile(negpath, negFolder(file).name,'*.jpg'))) == 0
        allFolders{file} = negFolder(file).name; 
    end
end
allFolders = allFolders(folderInds); % select randomly chosen folder inds
allFolders = allFolders(~cellfun('isempty',allFolders)); % remove empty folders from cell array 

%% Experiment Instructions
instructions = imread(instructpath);
imageTexture = Screen('MakeTexture', window, instructions);
Screen('DrawTexture', window, imageTexture, [], [], 0);
Screen(window,'Flip')

% captures keyboard until spacebar is pressed 
[~,~,keycode] = KbCheck; 
while (~keycode(spacebar)); 
    [~,~,keycode] = KbCheck; 
    WaitSecs(0.001); % ensures that loop doesn't hog CPU
    if find(keycode) == escape; % closes screen if escape key is pressed 
        Screen('Closeall')
    end
end;
while KbCheck;
end;

%% BLOCK LOOP %% 
for blockNumber = 1:numBlocks
      
    blockFolder = allFolders{blockNumber}; 
    
    % TEMPORARY: should reflect content of current blockFolder (ex., if all
    % images are dogs, object = 'dog' or some pointer to 'dog' in an array).
    object = 'object';
    
    for img = 0:23 
        posFiles{img+1} = fullfile(pospath,blockFolder,sprintf('%d.jpg',img));
        negFiles{img+1} = fullfile(negpath,blockFolder,sprintf('%d.jpg',img));
    end
    
    blockFiles = [posFiles(1:24)';negFiles(1:24)']; 
        
    % column1: imageID, column2: concept = 1 (positive) or 2 (negative), column3: correctKey = 188 (yes) or 190 (no) 
    blockSeq = [(1:48)',[ones(24,1);repmat(2,24,1)],[repmat(188,24,1);repmat(190,24,1)]]; 
    blockRandSeq = blockSeq(randperm(length(blockSeq)),:);
    
    %% Display block instructions (current visual concept task)     
    % TEMPORARY: simply displays category (folder name) and object filler 
    % (assuming content will be constant within any given block). 
    
    Screen('TextSize',window,textSize);
    [~,~,~] = DrawFormattedText(window,sprintf('Determine whether the [%s] is [%s].',object,blockFolder),'center',ycenter-75,black);
    [~,~,~] = DrawFormattedText(window,'Press < (left) if YES, or > (right) if NO','center','center',black);
    [~,~,~] = DrawFormattedText(window,'Press the spacebar to continue!','center',ycenter+75,black);
    Screen(window,'Flip');

    disp('current task: '); disp(blockFolder);
    
    % captures keyboard until spacebar is pressed 
    [~,~,keycode] = KbCheck; 
    while (~keycode(spacebar)); 
        [~,~,keycode] = KbCheck; 
        WaitSecs(0.001); % ensures that loop doesn't hog CPU
        if find(keycode) == escape; % closes screen if escape key is pressed 
            Screen('Closeall')
        end
    end;
    while KbCheck;
    end;

    % once spacebar is pressed, display text and move onto experiment
    [~,~,~] = DrawFormattedText(window,'get ready!','center','center',white);
    Screen(window,'Flip');
    WaitSecs(3); 
    
    % ------------ save block parameters in 'results' struct ------------ %      
    % content (noun) and concept (preposition/verb)
    results.blockContent{blockNumber,1} = object;
    results.blockConcept{blockNumber,1} = blockFolder;    
    
    % block sequence (order of presented trials) and correct answers
        % column1 = trial number, column2 = block number, column3 = imageID, column4 = pos/neg, column5 = correct key
        % plane = block number 
    results.blockSequence(:,:,blockNumber) = [(1:numTrials)',repmat(blockNumber,numTrials,1),blockRandSeq]; 
    
    % filenames of images used, in the order they were presented 
    results.blockFileIDs(:,blockNumber) = blockFiles(blockRandSeq(:,1));   
    % ------------------------------------------------------------------- %
    
    %% TRIAL LOOP %% 
    for trialNumber = 1:numTrials

        % skip non-existent pictures (in case files downloaded incorrectly)
        currfile = blockFiles{blockRandSeq(trialNumber,1)};
        if exist(currfile,'file') ~= 0
        
            % load current image and prepare on back buffer 
            trialImg = imread(blockFiles{blockRandSeq(trialNumber,1)});
            disp('positive = 1, negative = 2: ');disp(blockRandSeq(trialNumber,2));
            imageTexture = Screen('MakeTexture', window, trialImg);
            Screen('DrawTexture', window, imageTexture, [], [], 0);

            % prepare text on back buffer 
            Screen('TextSize',window,textSize);
            [~,~,~] = DrawFormattedText(window,'yes',yesLocation(1),yesLocation(2),black);
            [~,~,~] = DrawFormattedText(window,'no',noLocation(1),noLocation(2),black); 

            % show trial stimulus and text
            Screen(window,'Flip');
            WaitSecs(stimDuration);

            % record start time 
            start_time = GetSecs; 
            [~,secs,keycode] = KbCheck; % secs = time passed since start of KbCheck 

            Screen('FillRect', window, white);
            flipped = 0; % 0 = response window hasn't passed yet (no fixation cross), 1 = flipped to fixation cross 

            while ((isempty(find(keycode(respKeys))) || length(find(keycode))>1) && (secs-start_time)<respWindow);                                  % Waits for subject to respond with either of the appropriate keys
                [~,secs,keycode] = KbCheck;
                WaitSecs(0.001);
                
                % close everything if the escape key is pressed 
                if find(keycode)==escape; 
                    Screen('Closeall')
                end
                
                % flip to blank screen, if stimDuration has passed 
                if ~flipped && (secs-start_time >= stimDuration) 
                    Screen(window,'Flip');
                    flipped = 1;
                end 
            end

            % if key is pressed before stimDuration has passed, flip screen
            if flipped == 0  
                Screen(window,'Flip');
            end

            % determine response accuracy and present feedback 
            responseMade = find(keycode); 
            if ~isempty(responseMade)  
                responseTime=(secs-start_time)*1000; 
                if blockRandSeq(trialNumber,3) == responseMade 
                    responseScore = 1;
                    Screen('TextSize',window,textSize)
                    [~,~,~] = DrawFormattedText(window,'Correct!','center','center',[17,174,59]); 
                else 
                    responseScore = 0;
                    disp('incorrect')
                    [~,~,~] = DrawFormattedText(window,'Incorrect!','center','center',[228,25,25]);                 
                end
            elseif isempty(responseMade) 
                responseMade = 0;
                responseTime = 0;
                responseScore = 0;
                [~,~,~] = DrawFormattedText(window,'Respond Faster!','center','center',[228,25,25]);           
            end

            Screen(window,'Flip');
            WaitSecs(fixDuration); 

            % --------- save trial responses in 'results' struct -------- %          
            % row = trial, column = block
            results.correctResponses(trialNumber,blockNumber) = blockRandSeq(trialNumber,3);
            results.responsesMade(trialNumber,blockNumber) = responseMade;
            results.responseScores(trialNumber,blockNumber) = responseScore;
            results.responseTimes(trialNumber,blockNumber) = responseTime;           
            % ----------------------------------------------------------- %
            
        end
    end

%% End-of-block transition

    % if the current block isn't the last block, don't go to final screen yet
    if blockNumber < numBlocks 
        Screen('TextSize',window,textSize)
        [~,~,~] = DrawFormattedText(window,sprintf('End of block %d',blockNumber),'center',ycenter-100,black); 
        
        % display mean response accuracy for block 
        percentCorrect = round(nanmean(results.responseScores(:,blockNumber))*100);
        [~,~,~] = DrawFormattedText(window,sprintf('You answered %d%% percent of trials correctly.',percentCorrect),'center','center',black);
        if percentCorrect < 55 
            [~,~,~] = DrawFormattedText(window,'Please try your best next time!','center',ycenter+75,black);
        else 
            [~,~,~] = DrawFormattedText(window,'Good job!','center',ycenter+100,black);
        end      
        
        [~,~,~] = DrawFormattedText(window,'Press spacebar to go to the next block.','center',ycenter+200,black);
        Screen(window,'Flip');
        WaitSecs(1.5);

        % captures keyboard until spacebar is pressed 
        [~,~,keycode] = KbCheck; 
        while (~keycode(spacebar)); 
            [~,~,keycode] = KbCheck; 
            WaitSecs(0.001); % ensures that the loop doesn't hog the CPU
            if find(keycode) == escape; % closes screen if escape key is pressed 
                Screen('Closeall')
            end
        end;
        while KbCheck;
        end;       
        
        % once spacebar is pressed, move on to next block 
    end
end

%% END OF EXPERIMENT %% 
Screen('TextSize',window,textSize);
[~,~,~] = DrawFormattedText(window,'Great job, you finished! Thank you!','center','center',black); 
Screen(window,'Flip');
WaitSecs(1.5);

% -- EDIT FILEPATH AS NECESSARY (saves 'results' struct to file) -------- %
save(fullfile('...','data', sprintf('%s.mat',results.subjectID)),'-struct','results');
% ----------------------------------------------------------------------- %

Screen('Closeall') 
