function progressbar(c,t)
% This function creates a text progress bar. It should be called with a
% STRING argument to initialize and terminate. Otherwise the number correspoding
% to progress in % should be supplied.
% INPUTS:   C   Either: Text string to initialize or terminate
%                       Percentage number to show progress
% OUTPUTS:  N/A
% Example:  Please refer to demo_textprogressbar.m

% Author: Paul Proteus (e-mail: proteus.paul@yahoo.com)
% Version: 1.0 - textprogressbar(c)
% Changes tracker:  29.06.2010  - First version

% Inspired by: http://blogs.mathworks.com/loren/2007/08/01/monitoring-progress-of-a-calculation/

% Adapted by: Vitor Borges Santos (e-mail: borgessv93@gmail.com)
% Date: 23/03/2022

%% Initialization
persistent strCR;           %   Carriage return pesistent variable

% Vizualization parameters
% strPercentageLength = 20;   %   Length of percentage string (must be >5)
if ispc 
    strDotsMaximum = 50;   %   The total number of dots in a progress bar
else
    strDotsMaximum = 25;
end

%% Main

% if isempty(strCR) && ~ischar(c)
%     % Progress bar must be initialized with a string
%     % error('The text progress must be initialized with a string');
% else
if isempty(strCR) && ischar(c)
    % Progress bar - initialization
    fprintf('%s ',c);
    strCR = -1;
elseif ~isempty(strCR) && ischar(c)
    % Progress bar  - termination
    strCR = [];
    fprintf([c '\n']);
elseif isnumeric(c) && c ~= 100
    % Progress bar - normal progress
    percentageOut = [num2str(c,'%.2f') '%%'];
    if exist('t','var')
        [h,m,s] = hms(duration(0,0,t));
        percentageOut = [percentageOut ' done  --  time remaining: ' num2str(h) 'h ' num2str(m) 'm ' num2str(floor(s)) 's'];
    else
        percentageOut = [percentageOut ' done'];
    end
    nDots = floor(c/100*strDotsMaximum);
    if ispc
        dotOut = [char(hex2dec('258F')) repmat(char(hex2dec('258E')),1,nDots) repmat(char(hex2dec('2005')),1,strDotsMaximum-nDots) char(hex2dec('258F')) ' '];
    else
        dotOut = [char(hex2dec('2595')) repmat(char(hex2dec('258A')),1,nDots) repmat(char(hex2dec('2005')),1,strDotsMaximum-nDots) char(hex2dec('258F'))];
    end
    strOut = [dotOut percentageOut];

    % Print it on the screen
    if strCR == -1
        % Don't do carriage return during first run
        fprintf(strOut);
    else
        % Do it during all the other runs
        fprintf([strCR strOut]);
    end

    % Update carriage return
    strCR = repmat('\b',1,length(strOut)-1);
elseif isnumeric(c) && c == 100
    % Progress bar - normal progress
    %c = floor(c);
    if exist('t','var')
        percentageOut = [num2str(c,'%.0f') '%%'];
        [h,m,s] = hms(duration(0,0,t));
        percentageOut = [percentageOut ' done  --  total time: ' num2str(h) 'h ' num2str(m) 'm ' num2str(floor(s)) 's \n'];
        nDots = floor(c/100*strDotsMaximum);
        if ispc
            dotOut = [char(hex2dec('258F')) repmat(char(hex2dec('258E')),1,nDots) repmat(char(hex2dec('2005')),1,strDotsMaximum-nDots) char(hex2dec('258F')) ' '];
        else
            dotOut = [char(hex2dec('2595')) repmat(char(hex2dec('258A')),1,nDots) repmat(char(hex2dec('2005')),1,strDotsMaximum-nDots) ' '];
        end
        strOut = [dotOut percentageOut];
    else
        percentageOut = [num2str(c,'%.0f') '%%'];
        percentageOut = [percentageOut ' done \n'];
        nDots = floor(c/100*strDotsMaximum);
        if ispc
            dotOut = [char(hex2dec('258F')) repmat(char(hex2dec('258E')),1,nDots) repmat(char(hex2dec('2005')),1,strDotsMaximum-nDots) char(hex2dec('258F')) ' '];
        else
            dotOut = [char(hex2dec('2595')) repmat(char(hex2dec('258A')),1,nDots) repmat(char(hex2dec('2005')),1,strDotsMaximum-nDots) ' '];
        end
        strOut = [dotOut percentageOut];
    end
    % Print it on the screen
    fprintf([strCR strOut]);

    % Update carriage return
    strCR = [];

else
    % Any other unexpected input
    error('Unsupported argument type');
end