import React, { useState } from 'react';
import {
  BookOpen,
  CheckCircle,
  Code,
  Lightbulb,
  Award,
  ArrowRight,
  ChevronRight,
  Play,
  RotateCcw,
  Sparkles,
  HelpCircle,
} from 'lucide-react';
import { CurriculumLesson } from '../types/ml';
import { CURRICULUM_MODULES, CURRICULUM_LESSONS } from '../ml/curriculum';

interface LearnMLProps {
  currentLessonId?: string;
  completedLessons: string[];
  onToggleCompleteLesson: (lessonId: string) => void;
  onLaunchTask: (taskDesc: string) => void;
  onAskTutor: (context: any) => void;
}

export const LearnML: React.FC<LearnMLProps> = ({
  currentLessonId = CURRICULUM_LESSONS[0].id,
  completedLessons,
  onToggleCompleteLesson,
  onLaunchTask,
  onAskTutor,
}) => {
  const [selectedLessonId, setSelectedLessonId] = useState<string>(currentLessonId);
  const [selectedAnswers, setSelectedAnswers] = useState<Record<string, number>>({});
  const [showAnswerResults, setShowAnswerResults] = useState<Record<string, boolean>>({});

  // Interactive Mini-Demo State for the lesson (e.g., Bias-Variance Tradeoff slider)
  const [complexitySlider, setComplexitySlider] = useState<number>(3); // 1 = High Bias, 5 = High Variance

  const lesson =
    CURRICULUM_LESSONS.find((l) => l.id === selectedLessonId) ||
    CURRICULUM_LESSONS[0];

  const isCompleted = completedLessons.includes(lesson.id);

  const handleSelectOption = (questionId: string, optionIdx: number) => {
    setSelectedAnswers((prev) => ({ ...prev, [questionId]: optionIdx }));
    setShowAnswerResults((prev) => ({ ...prev, [questionId]: true }));
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-4 gap-8 animate-fadeIn">
      {/* Sidebar: Modules & Lesson Navigation (1 col) */}
      <div className="lg:col-span-1 space-y-6">
        <div>
          <div className="text-xs font-semibold text-indigo-400 uppercase tracking-wider">
            Course Curriculum
          </div>
          <h2 className="text-lg font-bold text-white mt-1">
            Zero to Practical Understanding
          </h2>
          <div className="text-xs text-slate-400 mt-1">
            {completedLessons.length} of {CURRICULUM_LESSONS.length} lessons completed
          </div>
        </div>

        <div className="space-y-4">
          {CURRICULUM_MODULES.map((mod) => {
            const moduleLessons = CURRICULUM_LESSONS.filter((l) => l.moduleId === mod.id);
            return (
              <div key={mod.id} className="space-y-1.5">
                <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider px-2">
                  Module {mod.id}: {mod.title}
                </div>

                <div className="space-y-1">
                  {moduleLessons.map((l) => {
                    const isDone = completedLessons.includes(l.id);
                    const isCurrent = l.id === lesson.id;
                    return (
                      <button
                        key={l.id}
                        onClick={() => setSelectedLessonId(l.id)}
                        className={`w-full text-left p-2.5 rounded-lg text-xs flex items-center justify-between transition-colors ${
                          isCurrent
                            ? 'bg-indigo-600 text-white font-semibold'
                            : 'bg-slate-900/60 hover:bg-slate-800 text-slate-300'
                        }`}
                      >
                        <span className="truncate pr-2">{l.title}</span>
                        {isDone ? (
                          <CheckCircle className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                        ) : (
                          <ChevronRight className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Main Content Area: Lesson Viewer (3 cols) */}
      <div className="lg:col-span-3 space-y-8">
        {/* Lesson Header */}
        <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-xs text-indigo-400 font-medium">
              <BookOpen className="w-4 h-4" />
              <span>Module {lesson.moduleId}: {lesson.moduleTitle}</span>
              <span className="text-slate-600">·</span>
              <span>Lesson {lesson.order}</span>
            </div>

            <button
              onClick={() => onToggleCompleteLesson(lesson.id)}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                isCompleted
                  ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                  : 'bg-slate-800 hover:bg-indigo-600 text-slate-300 hover:text-white'
              }`}
            >
              <CheckCircle className="w-3.5 h-3.5" />
              <span>{isCompleted ? 'Completed' : 'Mark as Completed'}</span>
            </button>
          </div>

          <h1 className="text-2xl md:text-3xl font-bold text-white tracking-tight">
            {lesson.title}
          </h1>

          <p className="text-sm text-slate-300 leading-relaxed max-w-3xl">
            {lesson.summary}
          </p>
        </div>

        {/* 1. Beginner Explanation & Technical Definition */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-2.5">
            <div className="flex items-center gap-2 text-xs font-bold text-emerald-400 uppercase tracking-wider">
              <Sparkles className="w-4 h-4" />
              <span>Intuitive Explanation (From Scratch)</span>
            </div>
            <p className="text-xs md:text-sm text-slate-200 leading-relaxed">
              {lesson.simpleExplanation}
            </p>
          </div>

          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-2.5">
            <div className="flex items-center gap-2 text-xs font-bold text-cyan-400 uppercase tracking-wider">
              <Lightbulb className="w-4 h-4" />
              <span>Technical & Mathematical Definition</span>
            </div>
            <p className="text-xs md:text-sm text-slate-300 leading-relaxed font-mono">
              {lesson.technicalDefinition}
            </p>
          </div>
        </div>

        {/* 2. Real World Example Card */}
        <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
          <div className="text-xs font-bold text-amber-400 uppercase tracking-wider">
            Real-World Industry Case Study
          </div>
          <p className="text-xs md:text-sm text-slate-300 leading-relaxed">
            {lesson.realWorldExample}
          </p>
        </div>

        {/* 3. Interactive Visual Demonstration Widget */}
        <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <div className="text-xs font-bold text-indigo-400 uppercase tracking-wider">
              Interactive Concept Sandbox: Model Complexity & Overfitting
            </div>
            <span className="text-xs text-slate-400 font-mono">
              Complexity Level: {complexitySlider} / 5
            </span>
          </div>

          <div className="w-full h-44 bg-slate-950 rounded-xl border border-slate-800/80 p-4 relative flex items-center justify-center overflow-hidden">
            <svg className="w-full h-full overflow-visible">
              {/* Noisy data points */}
              {[
                { x: 50, y: 110 }, { x: 100, y: 90 }, { x: 150, y: 75 },
                { x: 200, y: 80 }, { x: 250, y: 55 }, { x: 300, y: 65 },
                { x: 350, y: 40 }, { x: 400, y: 35 }, { x: 450, y: 20 },
              ].map((p, i) => (
                <circle key={i} cx={p.x} cy={p.y} r="4" fill="#6366f1" opacity="0.8" />
              ))}

              {/* Curve according to complexity slider */}
              {complexitySlider === 1 && (
                // Horizontal line (Underfit - High Bias)
                <line x1="30" y1="70" x2="480" y2="70" stroke="#f43f5e" strokeWidth="2.5" />
              )}
              {complexitySlider === 2 && (
                // Straight best fit line
                <line x1="30" y1="120" x2="480" y2="20" stroke="#10b981" strokeWidth="2.5" />
              )}
              {complexitySlider === 3 && (
                // Smooth balanced curve
                <path d="M 30,120 Q 250,80 480,20" fill="none" stroke="#10b981" strokeWidth="2.5" />
              )}
              {complexitySlider >= 4 && (
                // Overfitted wiggly curve hitting every noise point
                <path
                  d="M 30,120 L 50,110 L 100,90 L 150,75 L 200,80 L 250,55 L 300,65 L 350,40 L 400,35 L 450,20 L 480,15"
                  fill="none"
                  stroke="#f43f5e"
                  strokeWidth="2.5"
                />
              )}
            </svg>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-xs text-slate-400">Underfitting (High Bias)</span>
            <input
              type="range"
              min="1"
              max="5"
              value={complexitySlider}
              onChange={(e) => setComplexitySlider(Number(e.target.value))}
              className="flex-1 accent-indigo-500"
            />
            <span className="text-xs text-slate-400">Overfitting (High Variance)</span>
          </div>

          <p className="text-xs text-slate-400 text-center">
            {complexitySlider === 1 && 'Model is too simple (High Bias). It fails to capture the true underlying downward trend.'}
            {(complexitySlider === 2 || complexitySlider === 3) && 'Optimal trade-off: The model learns the general trend without memorizing noise.'}
            {complexitySlider >= 4 && 'Severe Overfitting (High Variance): The line contorts wildly to touch every sample, failing to generalize.'}
          </p>
        </div>

        {/* 4. Step-by-Step Practical Walkthrough */}
        <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-3">
          <div className="text-xs font-bold text-indigo-400 uppercase tracking-wider">
            Step-by-Step Procedure
          </div>
          <ol className="space-y-2 text-xs md:text-sm text-slate-300">
            {lesson.stepByStep.map((step, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-indigo-400 font-bold shrink-0">{idx + 1}.</span>
                <span>{step.replace(/^\d+\.\s*/, '')}</span>
              </li>
            ))}
          </ol>
        </div>

        {/* 5. Production Python & Scikit-Learn Code */}
        {lesson.codeSnippet && (
          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs font-bold text-slate-300">
                <Code className="w-4 h-4 text-indigo-400" />
                <span>Python & Scikit-Learn Implementation</span>
              </div>
            </div>
            <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 font-mono text-xs text-emerald-400 overflow-x-auto">
              <code>{lesson.codeSnippet}</code>
            </pre>
          </div>
        )}

        {/* 6. Practice Questions with Explanations */}
        <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
          <div className="text-xs font-bold text-indigo-400 uppercase tracking-wider">
            Practice Checkpoint Questions
          </div>

          <div className="space-y-6">
            {lesson.quizQuestions.map((q, qIdx) => {
              const selected = selectedAnswers[q.id];
              const isSubmitted = showAnswerResults[q.id];
              const isCorrect = selected === q.correctIndex;

              return (
                <div key={q.id} className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 space-y-3">
                  <div className="text-xs md:text-sm font-semibold text-white">
                    {qIdx + 1}. {q.question}
                  </div>

                  <div className="space-y-1.5">
                    {q.options.map((opt, optIdx) => (
                      <button
                        key={optIdx}
                        onClick={() => handleSelectOption(q.id, optIdx)}
                        className={`w-full text-left p-2.5 rounded-lg text-xs transition-colors border ${
                          selected === optIdx
                            ? isSubmitted && optIdx === q.correctIndex
                              ? 'bg-emerald-950 border-emerald-500 text-emerald-200'
                              : isSubmitted
                              ? 'bg-rose-950 border-rose-500 text-rose-200'
                              : 'bg-indigo-950 border-indigo-500 text-white'
                            : 'bg-slate-900/60 border-slate-800/80 text-slate-300 hover:border-slate-700'
                        }`}
                      >
                        {opt}
                      </button>
                    ))}
                  </div>

                  {isSubmitted && (
                    <div className={`p-3 rounded-lg text-xs space-y-1 ${isCorrect ? 'bg-emerald-950/60 text-emerald-300 border border-emerald-800/60' : 'bg-rose-950/60 text-rose-300 border border-rose-800/60'}`}>
                      <div className="font-semibold">
                        {isCorrect ? 'Correct!' : 'Incorrect'}
                      </div>
                      <p>{q.explanation}</p>
                      <p className="text-[11px] text-slate-400">
                        <strong className="text-slate-300">Common Misconception:</strong> {q.misconception}
                      </p>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* 7. Practical Task & Key Takeaways */}
        <div className="p-6 rounded-2xl bg-indigo-950/30 border border-indigo-900/40 space-y-4">
          <div className="text-xs font-bold text-indigo-400 uppercase tracking-wider">
            Practical Application Challenge
          </div>
          <p className="text-xs md:text-sm text-slate-300 leading-relaxed">
            {lesson.practicalTask}
          </p>

          <div className="pt-4 border-t border-indigo-900/40 space-y-2">
            <div className="text-xs font-bold text-white uppercase tracking-wider">
              Key Takeaways
            </div>
            <ul className="space-y-1 text-xs text-slate-300 list-disc list-inside">
              {lesson.keyTakeaways.map((takeaway, i) => (
                <li key={i}>{takeaway}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
