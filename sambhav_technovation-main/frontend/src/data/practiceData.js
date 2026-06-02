export const practiceGroups = [
    {
      id: "bp",
      title: "B vs P",
      subtitle: "Voicing contrast",
      focus: "Practice the difference between voiced /b/ and unvoiced /p/ sounds.",
      phonemes: ["b", "p"],
      lessons: [
        {
          id: "bp-initial",
          title: "Initial sounds",
          focus: "Hear the first sound clearly.",
          words: [
            { left: "bat", right: "pat" },
            { left: "big", right: "pig" },
            { left: "bin", right: "pin" },
            { left: "back", right: "pack" },
            { left: "ban", right: "pan" },
          ],
        },
        {
          id: "bp-final",
          title: "Final sounds",
          focus: "Hear the last sound clearly.",
          words: [
            { left: "tab", right: "tap" },
            { left: "slab", right: "slap" },
            { left: "rib", right: "rip" },
            { left: "mob", right: "mop" },
            { left: "cub", right: "cup" },
          ],
        },
      ],
    },
    {
      id: "rl",
      title: "R vs L",
      subtitle: "Tongue-shape contrast",
      focus: "Practice the difference between /r/ and /l/ sounds.",
      phonemes: ["r", "l"],
      lessons: [
        {
          id: "rl-initial",
          title: "Initial sounds",
          focus: "Listen for the first sound.",
          words: [
            { left: "right", right: "light" },
            { left: "road", right: "load" },
            { left: "rock", right: "lock" },
            { left: "red", right: "led" },
            { left: "rice", right: "lice" },
          ],
        },
        {
          id: "rl-final",
          title: "Final sounds",
          focus: "Listen for the end sound.",
          words: [
            { left: "car", right: "cal" },
            { left: "star", right: "stall" },
            { left: "far", right: "fall" },
          ],
        },
      ],
    },
    {
      id: "th",
      title: "TH Sounds",
      subtitle: "Tongue and airflow",
      focus: "Practice the voiceless /th/ and voiced /dh/ sounds.",
      phonemes: ["th", "dh"],
      lessons: [
        {
          id: "th-voiceless",
          title: "Voiceless TH",
          focus: "Let air flow without voice.",
          words: [
            { left: "think", right: "sink" },
            { left: "thin", right: "sin" },
            { left: "thumb", right: "some" },
            { left: "three", right: "tree" },
            { left: "bath", right: "bass" },
          ],
        },
        {
          id: "th-voiced",
          title: "Voiced TH",
          focus: "Use your voice with the tongue between the teeth.",
          words: [
            { left: "this", right: "dis" },
            { left: "that", right: "dat" },
            { left: "then", right: "den" },
            { left: "those", right: "doze" },
            { left: "the", right: "duh" },
          ],
        },
      ],
    },
    {
      id: "kg",
      title: "K vs G",
      subtitle: "Back-of-mouth contrast",
      focus: "Practice the difference between unvoiced /k/ and voiced /g/ sounds.",
      phonemes: ["k", "g"],
      lessons: [
        {
          id: "kg-initial",
          title: "Initial sounds",
          focus: "Listen for the back consonant at the start.",
          words: [
            { left: "coat", right: "goat" },
            { left: "cap", right: "gap" },
            { left: "came", right: "game" },
            { left: "cold", right: "gold" },
            { left: "cane", right: "gain" },
          ],
        },
      ],
    },
  ]
  
  export function getPracticeGroupById(groupId) {
    return practiceGroups.find((group) => group.id === groupId) || null
  }
  
  export function getPracticeLessonById(groupId, lessonId) {
    const group = getPracticeGroupById(groupId)
    if (!group) return null
  
    return group.lessons.find((lesson) => lesson.id === lessonId) || group.lessons[0] || null
  }