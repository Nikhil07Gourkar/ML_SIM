import React, { useState } from 'react';
import {
  Award,
  CheckCircle,
  HelpCircle,
  Play,
  RotateCcw,
  Sparkles,
  Trophy,
  ShieldCheck,
  Search,
  Lock,
} from 'lucide-react';
import { COMPREHENSIVE_QUIZ_QUESTIONS, PRACTICAL_CHALLENGES, ALL_BADGES } from '../ml/challenges';
import { UserProgress } from '../types/ml';

interface QuizChallengesProps {
  progress: UserProgress;
  onUnlockBadge: (badgeId: string) => void;
  onLaunchChallenge: (datasetId: string, algoId: any) => void;
}

export const QuizChallenges: React.FC<QuizChallengesProps> = ({
  progress,
  onUnlockBadge,
  onLaunchChallenge,
}) => {
  const [activeTab, setActiveTab] = useState<'quiz' | 'challenges' | 'badges'>('quiz');
  const [filterDifficulty, setFilterDifficulty] = useState<string>('All');

  // Quiz state
  const [currentQIdx, setCurrentQIdx] = useState<number>(0);
  const [userAnswers, setUserAnswers] = useState<Record<number, number>>({});
  const [isQuizSubmitted, setIsQuizSubmitted] = useState<boolean>(false);

  const filteredQuestions = COMPREHENSIVE_QUIZ_QUESTIONS.filter((q) =>
    filterDifficulty === 'All' ? true : q.difficulty === filterDifficulty
  );

  const activeQuestion = filteredQuestions[currentQIdx] || filteredQuestions[0];

  const handleSelectAnswer = (optIdx: number) => {
    if (isQuizSubmitted) return;
    setUserAnswers({ ...userAnswers, [currentQIdx]: optIdx });
  };

  const calculateScore = () => {
    let score = 0;
    filteredQuestions.forEach((q, idx) => {
      if (userAnswers[idx] === q.correctIndex) score++;
    });
    return score;
  };

  const handleFinishQuiz = () => {
    setIsQuizSubmitted(true);
    const score = calculateScore();
    if (score === filteredQuestions.length) {
      onUnlockBadge('Quiz Master');
    }
  };

  const handleResetQuiz = () => {
    setUserAnswers({});
    setIsQuizSubmitted(false);
    setCurrentQIdx(0);
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 text-xs font-medium text-amber-400">
            <Award className="w-4 h-4" />
            <span>Assessments & Real-World Machine Learning Challenges</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Quizzes, Challenges & Achievements
          </h1>
        </div>

        {/* Tab switch */}
        <div className="flex items-center gap-1.5 p-1 bg-slate-900 rounded-lg border border-slate-800">
          <button
            onClick={() => setActiveTab('quiz')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
              activeTab === 'quiz' ? 'bg-indigo-600 text-white font-semibold' : 'text-slate-400 hover:text-white'
            }`}
          >
            Concept Quiz
          </button>
          <button
            onClick={() => setActiveTab('challenges')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
              activeTab === 'challenges' ? 'bg-indigo-600 text-white font-semibold' : 'text-slate-400 hover:text-white'
            }`}
          >
            Practical Challenges
          </button>
          <button
            onClick={() => setActiveTab('badges')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
              activeTab === 'badges' ? 'bg-indigo-600 text-white font-semibold' : 'text-slate-400 hover:text-white'
            }`}
          >
            Earned Badges ({progress.earnedBadges.length})
          </button>
        </div>
      </div>

      {/* Tab 1: Concept Quiz */}
      {activeTab === 'quiz' && (
        <div className="max-w-3xl mx-auto space-y-6">
          {/* Difficulty Filter */}
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-2">
              <span className="text-slate-400">Difficulty:</span>
              {['All', 'Beginner', 'Intermediate', 'Advanced'].map((diff) => (
                <button
                  key={diff}
                  onClick={() => {
                    setFilterDifficulty(diff);
                    handleResetQuiz();
                  }}
                  className={`px-2.5 py-1 rounded-md transition-colors ${
                    filterDifficulty === diff
                      ? 'bg-slate-800 text-indigo-400 font-semibold'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  {diff}
                </button>
              ))}
            </div>

            <span className="text-slate-400 font-mono">
              Question {currentQIdx + 1} of {filteredQuestions.length}
            </span>
          </div>

          {/* Question Card */}
          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-6">
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <span className="text-indigo-400 font-medium">{activeQuestion.category}</span>
                <span>·</span>
                <span>{activeQuestion.difficulty}</span>
                <span>·</span>
                <span className="capitalize">{activeQuestion.type.replace('_', ' ')}</span>
              </div>
              <h2 className="text-lg font-bold text-white leading-relaxed">
                {activeQuestion.question}
              </h2>
            </div>

            {/* Options */}
            <div className="space-y-2.5">
              {activeQuestion.options.map((opt, oIdx) => {
                const isSelected = userAnswers[currentQIdx] === oIdx;
                const isCorrect = oIdx === activeQuestion.correctIndex;

                let optClass = 'bg-slate-950/60 border-slate-800/80 text-slate-300 hover:border-slate-700';
                if (isSelected) {
                  optClass = 'bg-indigo-950/80 border-indigo-500 text-white font-medium';
                }
                if (isQuizSubmitted) {
                  if (isCorrect) optClass = 'bg-emerald-950/80 border-emerald-500 text-emerald-200 font-bold';
                  else if (isSelected && !isCorrect) optClass = 'bg-rose-950/80 border-rose-500 text-rose-200';
                }

                return (
                  <button
                    key={oIdx}
                    onClick={() => handleSelectAnswer(oIdx)}
                    className={`w-full text-left p-3.5 rounded-xl text-xs transition-colors border ${optClass}`}
                  >
                    <div className="flex items-center gap-3">
                      <span className="w-5 h-5 rounded-full border border-slate-700 flex items-center justify-center text-[10px] shrink-0">
                        {String.fromCharCode(65 + oIdx)}
                      </span>
                      <span>{opt}</span>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Explanation when submitted */}
            {isQuizSubmitted && (
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs space-y-2">
                <div className="font-bold text-indigo-400">Answer Explanation</div>
                <p className="text-slate-300 leading-relaxed">{activeQuestion.explanation}</p>
                <div className="text-slate-400 pt-1 border-t border-slate-800 text-[11px]">
                  <strong className="text-slate-300">Common Misconception: </strong>
                  {activeQuestion.misconception}
                </div>
              </div>
            )}

            {/* Navigation buttons */}
            <div className="flex items-center justify-between pt-4 border-t border-slate-800">
              <button
                disabled={currentQIdx <= 0}
                onClick={() => setCurrentQIdx((i) => Math.max(0, i - 1))}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300 rounded-lg text-xs font-medium transition-colors"
              >
                Previous
              </button>

              <div className="flex items-center gap-2">
                {currentQIdx < filteredQuestions.length - 1 ? (
                  <button
                    onClick={() => setCurrentQIdx((i) => Math.min(filteredQuestions.length - 1, i + 1))}
                    className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold transition-colors"
                  >
                    Next Question
                  </button>
                ) : !isQuizSubmitted ? (
                  <button
                    onClick={handleFinishQuiz}
                    className="px-5 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-semibold transition-colors"
                  >
                    Submit Quiz
                  </button>
                ) : (
                  <button
                    onClick={handleResetQuiz}
                    className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    <span>Retake Quiz</span>
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Quiz Score Summary Banner */}
          {isQuizSubmitted && (
            <div className="p-6 rounded-2xl bg-indigo-950/40 border border-indigo-900/60 text-center space-y-2 animate-fadeIn">
              <Trophy className="w-8 h-8 text-amber-400 mx-auto" />
              <h3 className="text-lg font-bold text-white">
                Quiz Score: {calculateScore()} / {filteredQuestions.length} (
                {Math.round((calculateScore() / filteredQuestions.length) * 100)}%)
              </h3>
              <p className="text-xs text-slate-400">
                Review the explanations above to solidify concepts before running simulations!
              </p>
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Practical Challenges */}
      {activeTab === 'challenges' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {PRACTICAL_CHALLENGES.map((chal) => {
            const isCompleted = progress.completedChallenges.includes(chal.id);
            return (
              <div
                key={chal.id}
                className="p-6 rounded-2xl bg-slate-900 border border-slate-800 hover:border-slate-700 transition-all flex flex-col justify-between space-y-4"
              >
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span className="font-medium text-amber-400">{chal.difficulty}</span>
                    <span className="capitalize text-slate-500">Target: {chal.targetMetricName} ≥ {chal.targetMetricThreshold}</span>
                  </div>

                  <h3 className="text-base font-bold text-white">
                    {chal.title}
                  </h3>

                  <p className="text-xs text-slate-300 leading-relaxed">
                    {chal.description}
                  </p>

                  <div className="p-3 bg-slate-950 rounded-xl border border-slate-800/80 text-xs text-slate-400">
                    <strong className="text-indigo-400">Hint: </strong>
                    {chal.hint}
                  </div>
                </div>

                <div className="pt-4 border-t border-slate-800 flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs text-slate-400">
                    <Award className="w-4 h-4 text-amber-400" />
                    <span>Unlocks: &quot;{chal.badgeName}&quot;</span>
                  </div>

                  <button
                    onClick={() => onLaunchChallenge(chal.datasetId, chal.recommendedAlgorithm)}
                    className="px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>Launch Challenge</span>
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Tab 3: Earned Badges Showcase */}
      {activeTab === 'badges' && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {ALL_BADGES.map((b) => {
            const isEarned = progress.earnedBadges.some((eb) => eb.id === b.id || eb.title === b.title);
            return (
              <div
                key={b.id}
                className={`p-6 rounded-2xl border transition-all flex items-start gap-4 ${
                  isEarned
                    ? 'bg-slate-900 border-amber-500/40 text-white shadow-sm shadow-amber-500/10'
                    : 'bg-slate-950/60 border-slate-800/80 text-slate-500 opacity-60'
                }`}
              >
                <div
                  className={`p-3 rounded-xl ${
                    isEarned ? 'bg-amber-500/15 text-amber-400' : 'bg-slate-800 text-slate-600'
                  }`}
                >
                  {isEarned ? <Trophy className="w-6 h-6" /> : <Lock className="w-6 h-6" />}
                </div>

                <div className="space-y-1">
                  <div className="font-bold text-sm text-white">
                    {b.title}
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {b.description}
                  </p>
                  <div className="text-[10px] text-amber-400/80 font-mono mt-1">
                    {isEarned ? 'Unlocked' : 'Locked'}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
