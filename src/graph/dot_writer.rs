impl A {
    fn write_dot_representation(
        &mut self,
        submissions: &Vec<Submission>,
        writer: &mut dyn std::io::Write,
    ) -> std::io::Result<()> {
        let clusters = Fun::new(|w| {
            writeln!(w)?;

            // q0[label="Queue 0:"; peripheries=0; fontsize=15; fontname="Helvetica,Arial,sans-serif bold"];
            for q in 0..self.queues.len() {
                write!(
                    w,
                    r#"q{q}[label="{}:"; peripheries=0; fontsize=15; fontname="Helvetica,Arial,sans-serif bold"];"#,
                    self.get_queue_display(GraphQueue::new(q))
                        .set_prefix("Queue ")
                )?;
            }

            writeln!(w)?;

            let queue_submitions = |queue_index: usize| {
                submissions
                    .iter()
                    .filter(move |sub| sub.queue.index() == queue_index)
                    .flat_map(|sub| &sub.passes)
                    .cloned()
            };

            // subgraph cluster_0 {
            //     style=dashed;
            //     p0[label="#0"];
            //     p1[label="#1"];
            // }
            for (i, sub) in submissions.iter().enumerate() {
                let nodes = Fun::new(|w| {
                    for &p in &sub.passes {
                        write!(
                            w,
                            r#"p{}[label="{}"];"#,
                            p.index(),
                            self.get_pass_display(p).set_prefix("#")
                        )?;
                    }
                    Ok(())
                });

                token_abuse!(
                    w,
                    "subgraph cluster_$ {\
                        style=dashed;\
                        $\
                    }",
                    i,
                    nodes
                );
            }

            writeln!(w)?;

            // next edges will serve as layout constraints, don't show them
            write!(w, "edge[style=invis];")?;

            writeln!(w)?;

            let heads = (0..self.queues.len())
                .map(|q| queue_submitions(q).next())
                .collect::<Vec<_>>();

            // make sure that queues start vertically aligned
            if self.queues.len() > 1 {
                for q in 1..self.queues.len() {
                    write!(w, "q{} -> q{}[weight=99];", q - 1, q)?;
                }
            }

            // a weird way to count that heads has at least two Some(_)
            if heads.iter().flatten().nth(1).is_some() {
                let heads = heads
                    .iter()
                    .enumerate()
                    .filter_map(|(i, p)| p.as_ref().map(|p| (i, p)));

                for (q, p) in heads {
                    write!(w, "q{} -> p{};", q, p.index())?;
                }
            }

            writeln!(w)?;

            // p0 -> p1 -> p2;
            for q in 0..self.queues.len() {
                if queue_submitions(q).nth(1).is_some() {
                    let mut first = true;
                    for p in queue_submitions(q) {
                        if !first {
                            write!(w, " -> ")?;
                        }
                        write!(w, "p{}", p.index())?;
                        first = false;
                    }
                    write!(w, ";")?;
                }
            }

            writeln!(w)?;

            // { rank = same; q1; p2; p3; }
            for q in 0..self.queues.len() {
                let nodes = Fun::new(|w| {
                    write!(w, "q{q}; ")?;

                    for p in queue_submitions(q) {
                        write!(w, "p{}; ", p.index())?;
                    }
                    Ok(())
                });

                token_abuse!(w, (nodes)
                    {
                        rank = same;
                        $
                    }
                );
            }

            writeln!(w)?;
            write!(w, "edge[style=filled];")?;
            writeln!(w)?;

            // the visible edges for the actual dependencies
            for q in 0..self.queues.len() {
                for p in queue_submitions(q) {
                    let dependencies = &self.get_pass_data(p).dependencies;
                    for dep in dependencies {
                        write!(w, "p{} -> p{}", dep.index(), p.index())?;
                        if !dep.is_hard() {
                            write!(w, "[color=darkgray]")?;
                        }
                        write!(w, ";")?;
                    }
                }
            }

            Ok(())
        });
        let mut writer = WeirdFormatter::new(writer);
        token_abuse!(
            writer,
            (clusters)
            digraph G {
                fontname="Helvetica,Arial,sans-serif";
                node[fontname="Helvetica,Arial,sans-serif"; fontsize=9; shape=rect];
                edge[fontname="Helvetica,Arial,sans-serif"];

                newrank = true;
                rankdir = TB;

                $
            }
        );
        Ok(())
    }
    fn write_submissions_dot_representation(
        submissions: &Vec<Submission>,
        writer: &mut dyn std::io::Write,
    ) -> std::io::Result<()> {
        let clusters = Fun::new(|w| {
            for (i, sub) in submissions.iter().enumerate() {
                write!(w, r##"s{i}[label="{i} (q{})"];"##, sub.queue.index())?;
                if !sub.semaphore_dependencies.is_empty() {
                    write!(w, "{{")?;
                    for p in &sub.semaphore_dependencies {
                        write!(w, "s{} ", p.index())?;
                    }
                    write!(w, "}} -> s{i};")?;
                }
            }
            Ok(())
        });
        let mut writer = WeirdFormatter::new(writer);
        token_abuse!(
            writer,
            (clusters)
            digraph G {
                fontname="Helvetica,Arial,sans-serif";
                node[fontname="Helvetica,Arial,sans-serif"; fontsize=9; shape=rect];
                edge[fontname="Helvetica,Arial,sans-serif"];

                newrank = true;
                rankdir = TB;

                $
            }
        );
        Ok(())
    }
}
