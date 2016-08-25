%% Visual Concepts: Manual Image Check 

%% Startup & Subject Information 
sca; clc; clear; 
rng('default'); rng('shuffle');

% This is to ensure that the keycodes work correctly. If it still seems
% like the program isn't responding to your key presses, type "KbDemo" into
% the command window (after installing psychtoolbox) to check which
% keycodes match the spacebar, escape key, '<', and '>'. You can then
% manually enter those codes in the script later. 
computer = input('1 = mac, 2 = windows: ','s');

% Each of us gets a different 1/3 chunk of the images! 
subjectID = input('Enter your first initial (lowercase): ','s');

% 0 = small screen (for debugging), 1 = full screen (for experiment)
screenSize = input('Full Screen Mode (0/1): '); 

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
goodLocation = [xcenter-400,ycenter];
badLocation = [xcenter+400,ycenter];

[~, ~, ~] = DrawFormattedText(window, 'Preparing the task...' , 'center', 'center', black ); 
Screen(window, 'Flip');
WaitSecs(1);


%% Create three sets of tasks 

% -- EDIT FILEPATHS ----------------------------------------------------- % 
categoryFolders = '/.../images';
sampleCategory = '/.../images/backpack';
instructPath = '/.../check-instructions.jpg';
% ----------------------------------------------------------------------- % 

categoryDir = dir(categoryFolders);
numCategories = size(categoryDir,1)-2;

conceptDir = dir(sampleCategory);
numConcepts = size(conceptDir,1)-2;

start = 1;
for category = 1:numCategories
    tasks(start:start+(numConcepts-1),:) = [repmat(category,numConcepts,1),(1:numConcepts)'];
    start = start + (numConcepts-1);
end

% divide tasks into three sections (NOTE: the last one (phil's?) is a little 
% longer than the other two, but only by about ~2 minutes worth of trials)
if subjectID == 'c'
    tasks = tasks(1:numConcepts*9,:);
elseif subjectID == 't'
    tasks = tasks((numConcepts*9 + 1):(numConcepts*9 + numConcepts*9),:);
elseif subjectID == 'p'
    tasks = tasks((numConcepts*9 + numConcepts*9):length(tasks),:);
end

if computer == '1' % mac keycodes 
    respKeys = [54 55]; % 54 = < (yes), 55 = > (no) 
    spacebar = 44;
    escape = 41;
else % windows keycodes 
    respKeys = [188 190]; % 188 = < (yes), 190 = > (no) 
    spacebar = 32;
    escape = 27;
end

%% General Instructions
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
for block = 1:length(tasks)
    
    % get strings representing the current task's category and concept 
    currCategory = categoryDir(tasks(block,1)+2).name;
    currConcept = conceptDir(tasks(block,2)+2).name;
    
    % get strings representing five positive and five negative images within the current task
    % -- EDIT FILEPATHS ------------------------------------------------- %
    posPath = sprintf('/.../images/%s/%s/positive',currCategory,currConcept);
    posDir = dir(posPath);    
    negPath = sprintf('/.../images/%s/%s/negative',currCategory,currConcept);
    negDir = dir(negPath); 
    % ------------------------------------------------------------------- %
    
    % list filenames for all of the positive and negative images for this task
    for img = 1:5
        positiveFiles{img,1} = currPositive(img).name;
        negativeFiles{img,1} = currNegative(img).name;
    end
    
    % column1 lists all filenames for block, column2 = 1 if positive, 2 if negative
    allBlockFiles = [[positiveFiles;negativeFiles],[repmat(1,length(positiveFiles),1);repmat(2,length(negativeFiles),1)]];
    
    %% Display block instructions     
    
    Screen('TextSize',window,textSize);
    [~,~,~] = DrawFormattedText(window,sprintf('%s + %s',currCategory,currConcept),'center',ycenter-75,black);
    [~,~,~] = DrawFormattedText(window,'Press < (left) if GOOD, or > (right) if BAD','center','center',black);
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
    
    % ------------ save block parameters in 'results' cell array -------- %       
    results{:,3:4,block} = allBlockFiles; % filenames and positive/negative
    results{:,1,block} = currCategory; % category name 
    results{:,2,block} = currConcept; % concept name 
    results{:,5,block} = repmat((1:5)',2,1); % filenumbers 1-5 (.jpg)   
    % ------------------------------------------------------------------- %

    %% TRIAL LOOP %% 
    % one trial = one image (5 positive --> 5 negative)
    for image = 1:10 
        
        % load current image and prepare on back buffer 
        trialImgID = [];
        trialImg = imread(trialImgID);
        imageTexture = Screen('MakeTexture', window, trialImg);
        Screen('DrawTexture', window, imageTexture, [], [], 0);

        % prepare text on back buffer 
        Screen('TextSize',window,textSize);
        [~,~,~] = DrawFormattedText(window,'good',goodLocation(1),goodLocation(2),black);
        [~,~,~] = DrawFormattedText(window,'bad',badLocation(1),badLocation(2),black); 
        if image <= 5 % images 1:5 = positive
            [~,~,~] = DrawFormattedText(window,'[positive]',xcenter,ycenter-300,black);
        else % images 6:10 = negative
            [~,~,~] = DrawFormattedText(window,'[negative]',xcenter,ycenter-300,black);
        end
 
        % show trial stimulus and text
        Screen(window,'Flip');
        WaitSecs(stimDuration);

        Screen('FillRect', window, white);

        % waits for response (no time limit) 
        while ((isempty(find(keycode(respKeys))) || length(find(keycode))>1)); 
            [~,~,keycode] = KbCheck;
            WaitSecs(0.001);               
            % close everything if the escape key is pressed 
            if find(keycode)==escape; 
                Screen('Closeall')
            end          
        end
 
        % once key has been pressed, flip screen
        Screen(window,'Flip');

        % record response and present feedback 
        responseMade = find(keycode); 
        if ~isempty(responseMade)  
            if responseMade == respKeys(1)
                [~,~,~] = DrawFormattedText(window,'yay :)','center','center',[17,174,59]);
            elseif responseMade == respKeys(2)
                [~,~,~] = DrawFormattedText(window,'nay :(','center','center',[228,25,25]);
            end
        end

        Screen(window,'Flip');
        WaitSecs(0.5); 

        % --------- save trial responses in 'results' cell array -------- % 
        results{image,4,block} = trialImgID;
        results{image,5,block} = responseMade;
        % --------------------------------------------------------------- %        
    end   
end

%% END OF IMAGE CHECK %% 
Screen('TextSize',window,textSize);
[~,~,~] = DrawFormattedText(window,'Phew, all done! Great job, thank you!','center','center',black); 
Screen(window,'Flip');
WaitSecs(1.5);

% -- EDIT FILEPATH AS NECESSARY (saves 'results' struct to file) -------- %
save(fullfile('...','data', sprintf('%s.mat',subjectID)),'-struct','results');
% ----------------------------------------------------------------------- %

Screen('Closeall') 
