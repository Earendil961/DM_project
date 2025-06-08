\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{HSEtitle.cls}[Template for lab report]
%\LoadClass[a4paper]{extreport}
\LoadClass[fontsize=13pt]{scrreprt} % Times New Roman 13 size
\RequirePackage{ifpdf}
\RequirePackage{kvoptions}
\RequirePackage[a4paper,left=30mm,right=15mm,top=20mm,bottom=25mm,footskip=15mm]{geometry}
\RequirePackage{amsmath}

\RequirePackage{cmap}            % Улучшенный поиск русских слов в полученном pdf-файле
\RequirePackage[T2A]{fontenc}    % Поддержка русских букв
\RequirePackage[utf8]{inputenc}  % Кодировка utf8
\RequirePackage{csquotes}
\RequirePackage[main=russian,english]{babel}  % Языки: русский, английский

\RequirePackage{listings}
\RequirePackage{xcolor}

\RequirePackage[font=small,skip=4pt]{caption}
\RequirePackage{graphicx}
\RequirePackage{hyphenat} % чтобы не было переноса

% \usepackage{pscyr}    % Красивые русские шрифты
% \renewcommand{\rmdefault}{ftm} % Включаем Times New Roman

%%% Оформление абзацев %%%
\RequirePackage{indentfirst} % Красная строка

%%% Размер шрифатов у заголовков %%%
\RequirePackage{titlesec}

\sloppy   % Избавляемся от переполнений
\linespread{1.5} % Полуторный интервал
\setlength{\parindent}{12.5mm} % Отступ красной строки

\RequirePackage{float}
% Попись таблицы и рисунка
\captionsetup[table]{position=t, singlelinecheck=false, justification=raggedleft, labelsep=endash, name=Таблица}
\captionsetup[figure]{labelfont={it}, textfont={it}, position=b, singlelinecheck=false, labelsep=endash, justification=centering, name=Рисунок}

%%% Главы и разделы %%%
\titleformat{\chapter}%[display]
  {\Large\bfseries\centering}
  {\thechapter}{6pt}{}
\titlespacing*{\chapter}{0pt}{0pt}{12pt}
\titleformat{\section}
  {\large\bfseries\centering}
  {\thesection}{6pt}{}
\titlespacing*{\section}{0pt}{12pt}{6pt}
\titleformat{\subsection}
  {\bfseries\raggedright}
  {\thesubsection}{4pt}{}
\titlespacing*{\subsection}{0pt}{8pt}{4pt}
\titleformat{\subsubsection}
  {\itshape\raggedright}
  {\thesubsubsection}{4pt}{}
\titlespacing*{\subsubsection}{0pt}{8pt}{4pt}

%%% Маркированные списки %%%
\renewcommand\labelitemi{--}
\renewcommand\labelitemii{--}
\renewcommand\labelitemiii{--}
\renewcommand\labelitemiv{--}
\RequirePackage{enumitem}
\setlist[enumerate]{label={\arabic*.}, ref={\arabic*}, leftmargin={20mm}, nosep} % все уровни нумерованного списка арабские цифры
\setlist[itemize]{leftmargin={20mm}, nosep}

%%%%% Listing %%%%
%New colors defined below
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}
\definecolor{whitesmoke}{rgb}{0.96, 0.96, 0.96}

\lstdefinestyle{listingstyle}{
  backgroundcolor=\color{backcolour},
  commentstyle=\color{codegreen},
  keywordstyle=\color{magenta},
  numberstyle=\tiny\color{codegray},
  stringstyle=\color{codepurple},
  basicstyle=\ttfamily\footnotesize,
  breakatwhitespace=false,
  breaklines=true,
  captionpos=b,
  keepspaces=true,
  numbers=left,
  numbersep=5pt,
  showspaces=false,
  showstringspaces=false,
  showtabs=false,
  tabsize=4,
  inputencoding=utf8,
  extendedchars=true,
  literate=
    {а}{{\cyra}}1
    {б}{{\cyrb}}1
    {в}{{\cyrv}}1
    {г}{{\cyrg}}1
    {д}{{\cyrd}}1
    {е}{{\cyre}}1
    {ё}{\"{\cyre}}1
    {ж}{{\cyrzh}}1
    {з}{{\cyrz}}1
    {и}{{\cyri}}1
    {й}{{\cyrishrt}}1
    {к}{{\cyrk}}1
    {л}{{\cyrl}}1
    {м}{{\cyrm}}1
    {н}{{\cyrn}}1
    {о}{{\cyro}}1
    {п}{{\cyrp}}1
    {р}{{\cyrr}}1
    {с}{{\cyrs}}1
    {т}{{\cyrt}}1
    {у}{{\cyru}}1
    {ф}{{\cyrf}}1
    {х}{{\cyrh}}1
    {ц}{{\cyrc}}1
    {ч}{{\cyrch}}1
    {ш}{{\cyrsh}}1
    {щ}{{\cyrshch}}1
    {ъ}{{\cyrhrdsn}}1
    {ы}{{\cyrery}}1
    {ь}{{\cyrsftsn}}1
    {э}{{\cyrerev}}1
    {ю}{{\cyryu}}1
    {я}{{\cyrya}}1
    {А}{{\CYRA}}1
    {Б}{{\CYRB}}1
    {В}{{\CYRV}}1
    {Г}{{\CYRG}}1
    {Д}{{\CYR96}}1
    {Е}{{\CYRE}}1
    {Ё}{{\"{\CYRE}}}1
    {Ж}{{\CYRZH}}1
    {З}{{\CYRZ}}1
    {И}{{\CYRI}}1
    {Й}{{\CYRISHRT}}1
    {К}{{\CYRK}}1
    {Л}{{\CYRL}}1
    {М}{{\CYRM}}1
    {Н}{{\CYRN}}1
    {О}{{\CYRO}}1
    {П}{{\CYRP}}1
    {Р}{{\CYRR}}1
    {С}{{\CYRS}}1
    {Т}{{\CYRT}}1
    {У}{{\CYRU}}1
    {Ф}{{\CYRF}}1
    {Х}{{\CYRH}}1
    {Ц}{{\CYRC}}1
    {Ч}{{\CYRCH}}1
    {Ш}{{\CYRSH}}1
    {Щ}{{\CYRSHCH}}1
    {Ъ}{{\CYRHRDSN}}1
    {Ы}{{\CYRERY}}1
    {Ь}{{\CYRSFTSN}}1
    {Э}{{\CYREREV}}1
    {Ю}{{\CYRYU}}1
    {Я}{{\CYRYA}}1
}
\lstset{style=listingstyle} % для красивых листингов
%% Оформление подписей к листингам в стандарте вышки
\DeclareCaptionFormat{modifiedlst}{\lstlistingname~\thelstlisting~--~#3}
\captionsetup[lstlisting]{format=modifiedlst}

% Если язык не поддерживается (например, js) можно добавить свой https://tex.stackexchange.com/questions/89574/language-option-supported-in-listings
% или воспользоваться функцией без выделения синтаксиса через \lstset{style=listingstyle}
\lstdefinestyle{monochromestyle}{
  backgroundcolor=\color{whitesmoke},
  extendedchars=true,
  basicstyle=\footnotesize\ttfamily,
  showstringspaces=false,
  showspaces=false,
  numberstyle=\footnotesize,
  numbersep=9pt,
  breakatwhitespace=false,
  breaklines=true,
  captionpos=b,
  keepspaces=true,
  numbers=left,
  numbersep=5pt,
  showspaces=false,
  showstringspaces=false,
  showtabs=false,
  tabsize=4,
  inputencoding=utf8,
  extendedchars=true,
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Библиография %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[
  style=gost-numeric-min,
  language=auto,
  backend=biber,
  defernumbers=true,% Нумерация начинается как только встречается группа.
  sorting=none,
  url=false,
  doi=false,
  isbn=false,
  movenames=false,
  bibencoding=utf8,% кодировка bib файла
  language=autobib,% получение языка из babel/polyglossia, default: autobib % если ставить autocite или auto, то цитаты в тексте с указанием страницы, получат указание страницы на языке оригинала
  autolang=other,% многоязычная библиография
  clearlang=true,% внутренний сброс поля language, если он совпадает с языком из babel/polyglossia
  % defernumbers=true,% нумерация проставляется после двух компиляций, зато позволяет выцеплять библиографию по ключевым словам и нумеровать не из большего списка
  sortcites=true,% сортировать номера затекстовых ссылок при цитировании (если в квадратных скобках несколько ссылок, то отображаться будут отсортированно, а не абы как)
  maxnames=50,
]{biblatex}
\let\bibnamedelimd\addnbspace % имена с неразрывным пробелом tex.stackexchange.com/questions/567160

%%%%%%%% для форматирования под гост 2011
%%% https://github.com/AndreyAkinshin/Russian-Phd-LaTeX-Dissertation-Template/blob/dbd175fb59fe5d1d29df3048ce4aded671fb7db9/Readme/Bibliography.md#%D1%81%D1%82%D1%80%D0%BE%D0%B3%D0%BE%D1%81%D1%82%D1%8C-%D1%81%D0%BE%D0%BE%D1%82%D0%B2%D0%B5%D1%82%D1%81%D1%82%D0%B2%D0%B8%D1%8F-%D0%B3%D0%BE%D1%81%D1%82
\renewcommand*{\multicitedelim}{\addcomma\space}
\ltx@iffilelater{biblatex-gost.def}{2017/05/03}%
{\toggletrue{bbx:gostbibliography}%

\renewcommand*{\revsdnamepunct}{\addcomma}}{}
% Настройки языка в списке литературы.
\DeclareSourcemap{
  \maps[datatype=bibtex,overwrite=false]{
    \map{
      \step[fieldset=language,fieldvalue=english]
    }
    \map{
      \step[fieldsource=language,match=\regexp{English},replace=\regexp{english}]
    }
    \map{
      \step[fieldsource=language]
      \step[fieldset=hyphenation,origfieldval]
    }
  }
}
%%%%%%%%%%%%%%%%%%%%%%
% ошибка в russian.lbx
\NewBibliographyString{langjapanese}
\NewBibliographyString{fromjapanese}

% TODO change to printbibliography and \renewcommand{\bibname}{Whatever floats your boat} \renewcommand{\bibsection}{\section{\bibname}}
% \printbibliography[title={Articles},type=article,sorting=nyt,heading=subbibnumbered]
\newcommand{\putbibliography}{
  \chapter*{Список использованных источников}
  \addcontentsline{toc}{chapter}{Список использованных источников}
  \printbibliography[heading=none] %title={Список использованных источников}, env=gostbibliography,
}



%%%%%%%%%% 
\def\@year{\the\year{}}
\def\@professorTitle{}
\def\@professorAcademic{}
\def\@professorName{I.I.~Ivanov}
\def\@group{AA-00-0}
\def\@specName{Прикладной анализ данных и искусственный интеллект}
\def\@courseName{Алгоритмы и структуры данных}

\newcommand{\Group}[1]{\def\@group{#1}}
\newcommand{\Year}[1]{\def\@year{#1}}
\newcommand{\specName}[1]{\def\@specName{#1}}
\newcommand{\courseName}[1]{\def\@courseName{#1}}
\newcommand{\professor}[3]{
  \def\@professorTitle{#1}
  \def\@professorAcademic{#2}
  \def\@professorName{#3}
}


\newcommand\tline[3]{
  $\underset{\text{\tiny{(#3)}}}{\underline{\makebox[#1\textwidth][l]{#2}}}$
}

%%%%%%%%%% Титульный лист %%%%%%%%%%
\renewcommand{\maketitle}{
\thispagestyle{empty}
{\centering
{Правительство Российской Федерации\\\MakeUppercase{Федеральное государственное автономное\\образовательное учреждение высшего образования\\
<<Национальный исследовательский университет\\
<<Высшая школа экономики>>}\\
(НИУ ВШЭ --- Санкт-Петербург)}
\vspace{3mm}
\hrule
\vspace{5mm}
% \vfill
Санкт-Петербургская школа физико-математических и компьютерных наук\\
<<\@specName>>
\vfill
\vfill
{
    \raggedright
    \begin{minipage}{0.4\textwidth}
    Преподаватель: 
        \vspace{3mm}
        
        \nohyphens{\@professorTitle}, \nohyphens{\@professorAcademic}\\
        \@professorName
      % Преподаватель\\
      % \tline{1}{\@professorTitle, \@professorAcademic}{должность, ученая степень, ученое звание}\\
      % \tline{1}{\@professorName}{Фамилия, Имя, Отчество}\\
    \end{minipage}\hfill
}
\vfill
\vfill
\textbf{\MakeUppercase{Отчёт о лабораторной работе}}
\vfill
по курсу: <<\@courseName>>
\vfill
\vfill
\hfill\begin{minipage}[c]{0.5\textwidth}
    Выполнил студент группы~\underline{\makebox[0.37\textwidth][l]{\@group}}\\
    \tline{1}{\@author}{Фамилия, Имя, Отчество}\\
  \end{minipage}
\vfill
\vfill
\vfill
Санкт-Петербург, \@year{}
\newpage\par}
}
