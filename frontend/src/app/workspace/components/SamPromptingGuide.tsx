"use client";

import { useEffect, useState } from "react";
import { ArrowLeft, BookOpen, ChevronRight, X } from "lucide-react";
import styles from "./SamPromptingGuide.module.css";

type GuideTopic = "click" | "box" | "similar";

interface GuideSection {
  heading: string;
  paragraphs: string[];
}

interface GuideArticle {
  title: string;
  summary: string;
  sections: GuideSection[];
}

interface Props {
  initialTopic: GuideTopic;
  onClose: () => void;
}

const ARTICLES: Record<GuideTopic, GuideArticle> = {
  click: {
    title: "Point prompting",
    summary: "Use a point to ask SAM for a particle mask.",
    sections: [
      {
        heading: "When to use it",
        paragraphs: [
          "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer posuere, justo sit amet volutpat facilisis, neque nibh tincidunt lectus, vitae bibendum eros sapien nec purus.",
          "Suspendisse potenti. Mauris a nibh sed nulla commodo ultrices. Choose a visible particle with enough space around it when you place the prompt.",
        ],
      },
      {
        heading: "Place a prompt",
        paragraphs: [
          "Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae. Click once near the center of the particle. The proposed mask appears for review before it becomes part of the saved segmentation.",
        ],
      },
      {
        heading: "Review the result",
        paragraphs: [
          "Donec vitae sapien ut libero venenatis faucibus. Keep a proposal that follows the particle boundary. Discard it when it includes a neighbor or misses a large part of the particle.",
        ],
      },
    ],
  },
  box: {
    title: "Box prompting",
    summary: "Draw a box around a particle to guide the mask.",
    sections: [
      {
        heading: "When to use it",
        paragraphs: [
          "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Box prompts can help when particles are close together or a point prompt returns too much of a neighboring shape.",
        ],
      },
      {
        heading: "Draw a useful box",
        paragraphs: [
          "Curabitur ullamcorper ultricies nisi. Drag a tight rectangle around one particle. Leave a small margin so the box contains the full boundary, while excluding nearby particles where possible.",
        ],
      },
      {
        heading: "Review the result",
        paragraphs: [
          "Nam eget dui. The proposed mask is still a draft. Accept it to keep it with the pending proposals, or click the proposal to discard it.",
        ],
      },
    ],
  },
  similar: {
    title: "Similar particle search",
    summary: "Use existing particles to find more candidates in the image.",
    sections: [
      {
        heading: "Prepare examples",
        paragraphs: [
          "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Add a few clear particles first. These examples help SAM look for similar shapes elsewhere in the image.",
        ],
      },
      {
        heading: "Search the image",
        paragraphs: [
          "Aenean vulputate eleifend tellus. Start the search and review each proposed mask. Results are candidates; inspect their boundaries before accepting them.",
        ],
      },
      {
        heading: "Review candidates",
        paragraphs: [
          "Phasellus viverra nulla ut metus varius laoreet. Accept useful candidates and discard unwanted ones. The original segmentation remains available while you review proposals.",
        ],
      },
    ],
  },
};

const TOPIC_ORDER: GuideTopic[] = ["click", "box"];
const TOPIC_LABELS: Record<GuideTopic, string> = {
  click: "Point prompting",
  box: "Box prompting",
  similar: "Similar search",
};

export default function SamPromptingGuide({ initialTopic, onClose }: Props) {
  const [topic, setTopic] = useState<GuideTopic>(initialTopic);
  const article = ARTICLES[topic];

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <div className={styles.backdrop} onClick={onClose}>
      <section
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-label="SAM prompting guide"
        onClick={event => event.stopPropagation()}
      >
        <header className={styles.header}>
          <div className={styles.breadcrumb}>
            <BookOpen size={15} />
            <span>Help</span>
            <ChevronRight size={12} />
            <span>SAM prompting</span>
          </div>
          <button type="button" className={styles.closeButton} onClick={onClose} aria-label="Back to workspace">
            <ArrowLeft size={15} /> Back to workspace
            <X size={14} />
          </button>
        </header>

        <div className={styles.body}>
          <nav className={styles.navigation} aria-label="SAM prompting topics">
            <div className={styles.navigationTitle}>SAM PROMPTING</div>
            {TOPIC_ORDER.map(item => (
              <button
                key={item}
                type="button"
                className={`${styles.topicButton} ${topic === item ? styles.topicActive : ""}`}
                onClick={() => setTopic(item)}
              >
                {TOPIC_LABELS[item]}
                {topic === item && <ChevronRight size={13} />}
              </button>
            ))}
            <div className={styles.navigationNote}>Bundled with TEMseg</div>
          </nav>

          <article className={styles.article}>
            <div className={styles.placeholderBanner}>PLACEHOLDER COPY · UI PREVIEW</div>
            <h1>{article.title}</h1>
            <p className={styles.summary}>{article.summary}</p>
            <p className={styles.intro}>This guide opens inside the workspace. Close it to return to the same image and prompt state.</p>
            {article.sections.map(section => (
              <section className={styles.articleSection} key={section.heading}>
                <h2>{section.heading}</h2>
                {section.paragraphs.map(paragraph => <p key={paragraph}>{paragraph}</p>)}
              </section>
            ))}
            <div className={styles.footerNote}>Lorem ipsum is temporary. This content will be replaced with the SAM prompting guide.</div>
          </article>
        </div>
      </section>
    </div>
  );
}
