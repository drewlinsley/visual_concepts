%% Visual Concepts: Human 2AFC Behavioral Experiment 
% Deep Learning Hackathon 2016 

%% Startup & participant information 
sca; clc; clear; 
rng('default'); rng('shuffle'); % randomize the seed for pseudorandomisation

% prompt for subject information in command window and save responses in final data struct ('results'). 
computer = input('1 = mac, 2 = windows: ','s');
results.subjectID = input('Enter Subject ID (FIRSTLAST): ','s');
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
yesLocation = [xcenter-400,ycenter];
noLocation = [xcenter+400,ycenter];
white = WhiteIndex(window); % white pixel intensity 
black = BlackIndex(window); % black pixel intensity 
Screen('TextSize', window, 30); 
Screen('TextFont',window,'Helvetica');

[~, ~, ~] = DrawFormattedText(window, 'Preparing the task...' , 'center', 'center', black ); 
Screen(window, 'Flip');
WaitSecs(1);

%% Specify experiment parameters 

% -- CHECK WITH PERSONAL COMPUTER --------------------------------------- %
if computer == '1' % mac keycodes 
    respKeys = [54 55]; % 54 = < (yes), 55 = > (no) 
    spacebar = 44;
    escape = 41;
else % windows keycodes 
    respKeys = [188 190]; % 188 = < (yes), 190 = > (no) 
    spacebar = 32;
    escape = 27;
end
% ----------------------------------------------------------------------- %

% -- ADJUST AS NECESSARY ------------------------------------------------ %
% NOTE: respWindow & stimDuration don't have to be equal 
respWindow = 0.5; % seconds subject is given to respond
stimDuration = 0.5; % seconds image is on screen
fixDuration = 1; % seconds feedback/fixation cross is on screen 
% ----------------------------------------------------------------------- % 

% -- EDIT FILEPATHS ----------------------------------------------------- % 
categoryPath = '/.../final_images';
sampleconceptPath = '/.../final_images/bear';
instructPath = '/.../instructions.jpg';
% ----------------------------------------------------------------------- % 

addpath(categoryPath,sampleconceptPath,instructPath);

% numBlocks = # concepts, numTrials = # images per concept
% NOTE: a 'block' is a concept task (ex. 'is the dog within, y/n?')
numBlocks = 10; 
numTrials = 100; 

% -- SELECTS CATEGORY FOLDER -------------------------------------------- %
% if it's one of us, make sure we get a certain image set 
if results.subjectID == 'cf'
    category = 'horse';
elseif results.subjectID == 'pb'
    category = 'elephant';
elseif results.subjectID == 'tp'
    category = 'bear';
else
    % otherwise, randomly select a category folder (one per subject) 
    randcat = randi([1 3],1);
    if randcat == 1
        category = 'bear';
    elseif randcat == 2
        category = 'elephant';
    else 
        category = 'horse'; 
    end
end

% select first ten concept folders 
conceptDir = dir(sampleconceptPath);
for i = 1:numBlocks
    currConcept = conceptDir(i+2).name;
    concepts{i,1} = currConcept;
end
% ----------------------------------------------------------------------- %

% list 50 negative and 50 positive image filenames for each concept 
%   rows (numTrials) = image filenames (0.jpg - 49.jpg)
%   column1 = image filename, column2 = 1 if positive, -1 if negative; column3 = correct response key
%   planes (numBlocks) = concepts
for concept = 1:size(concepts,1)
    for i = 1:numTrials
        if i < 51 % first 50 listed = positive 
            allImages{i,1,concept} = sprintf('%s/%s/%s/positive/%s.jpg',categoryPath,category,concepts{1},num2str(i-1));
            allImages{i,2,concept} = 1;
            allImages{i,3,concept} = respKeys(1);
        else % last 50 listed = negative 
            allImages{i,1,concept} = sprintf('%s/%s/%s/negative/%s.jpg',categoryPath,category,concepts{1},num2str((i-50)-1));
            allImages{i,2,concept} = -1;
            allImages{i,3,concept} = respKeys(2);
        end        
    end
end

%% Experiment Instructions
instructions = imread(instructPath);
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
    
    % shuffle positive and negative images within current block context
    currBlockImages = allImages(randperm(size(allImages,1)),:,blockNumber);
    
    %% Display block instructions (current visual concept task)     
    
    [~,~,~] = DrawFormattedText(window,sprintf('%s + %s.',category,concepts{blockNumber,1}),'center',ycenter-75,[32,114,220]);
    [~,~,~] = DrawFormattedText(window,'Press < (left) if YES, or > (right) if NO','center','center',black);
    [~,~,~] = DrawFormattedText(window,'Press the spacebar to continue!','center',ycenter+75,black);
    Screen(window,'Flip');
    
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
    [~,~,~] = DrawFormattedText(window,'get ready!','center','center',black);
    Screen(window,'Flip');
    WaitSecs(2); 
    
    % ------------ save block parameters in 'results' struct ------------ %      
    % content (noun) and concept (preposition/verb)
    results.blockCategory{blockNumber,1} = category;
    results.blockConcept{blockNumber,1} = concepts;    
    
    % block sequence (order of presented trials) and correct answers
    % saved as nested cell arrays 
    results.blockSequence{blockNumber} = currBlockImages;  
    % ------------------------------------------------------------------- %
    
    %% TRIAL LOOP %% 
    for trialNumber = 1:numTrials

        % skip non-existent pictures (in case files downloaded incorrectly)        
         currfile = currBlockImages{trialNumber,1};
        
            % load current image and prepare on back buffer 
            trialImage = imread(currfile);
            imageTexture = Screen('MakeTexture', window, trialImage);
            Screen('DrawTexture', window, imageTexture, [], [], 0);

            % prepare text on back buffer 
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
                if currBlockImages{trialNumber,3} == responseMade 
                    responseScore = 1;
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
            results.correctResponses(trialNumber,blockNumber) = cell2mat(currBlockImages(trialNumber,3));
            results.responsesMade(trialNumber,blockNumber) = responseMade;
            results.responseScores(trialNumber,blockNumber) = responseScore;
            results.responseTimes(trialNumber,blockNumber) = responseTime;           
            % ----------------------------------------------------------- %
    end

%% End-of-block transition

    % if the current block isn't the last block, don't go to final screen yet
    if blockNumber < numBlocks 
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
[~,~,~] = DrawFormattedText(window,'Great job, you finished! Thank you!','center','center',black); 
Screen(window,'Flip');
WaitSecs(1.5);

% -- EDIT FILEPATH AS NECESSARY (saves 'results' struct to file) -------- %
save(sprintf('/.../%s',results.subjectID),'-struct','results')
% ----------------------------------------------------------------------- %

Screen('Closeall') 
