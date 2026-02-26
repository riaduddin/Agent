
import random

# Dynamic Status Messages for a better user experience
STATUS_MESSAGES = {
    "creating_outline": [
        "Mapping out your presentation structure...",
        "Crafting a custom outline for you...",
        "Brainstorming the perfect flow for your slides...",
        "Organizing your ideas into a slide-by-slide plan...",
        "Designing the blueprint for your presentation...",
        "Curating a tailored presentation flow just for you...",
        "Scripting the foundation of your slides..."
    ],
    "outline_created": [
        "Success! Your {total_slides}-slide outline is ready.",
        "Outline complete: {total_slides} slides mapped out.",
        "Great start! Plan for {total_slides} slides finalized.",
        "Boom! Your {total_slides}-slide structure is all set.",
        "Done! We've planned a {total_slides}-slide journey for you.",
        "Your presentation roadmap is ready - {total_slides} slides of potential!",
        "Outline locked in! {total_slides} slides ready for design."
    ],
    "generating_slides": [
        "Selecting the best templates and building your {total_slides} slides...",
        "Bringing your {total_slides} slides to life with premium designs...",
        "Sprucing things up! Generating {total_slides} slides for you...",
        "Creating {total_slides} visually stunning slides based on your plan...",
        "Magic is happening! Customizing {total_slides} template-driven slides...",
        "Hand-selecting layouts and styles for your {total_slides} slides...",
        "Applying the finishing touches to {total_slides} custom slides..."
    ],
    "slide_ready": [
        "Slide {idx}/{total} is prepped and ready for its closeup.",
        "Done! Slide {idx}/{total} is ready to be generated.",
        "Slide {idx}/{total} plan is solid and ready to go.",
        "Setting up: Slide {idx}/{total} is all prepped.",
        "Ready! Slide {idx}/{total} is moving to the next stage.",
        "Slide {idx}/{total} is ready for its transformation.",
        "All systems go for Slide {idx}/{total}!"
    ],
    "slide_fallback": [
        "Slide {idx}/{total} is taking a unique path (no template matched).",
        "Slide {idx}/{total}: Going custom since no exact template fit.",
        "Customizing: Slide {idx}/{total} using a smart fallback design.",
        "Adjusting: Slide {idx}/{total} will be unique (custom layout).",
        "Not using a template for Slide {idx}/{total}, but it'll still look great!",
        "Slide {idx}/{total} is getting a specialized custom treatment.",
        "No template for Slide {idx}/{total}, so we're crafting it from scratch!"
    ],
    "slide_generated": [
        "Just finished Slide {count}/{total}!",
        "Slide {count}/{total} has been brought to life.",
        "Another one! Slide {count}/{total} is ready.",
        "Moving fast: Slide {count}/{total} is complete.",
        "Success: Slide {count}/{total} is now part of your deck.",
        "Slide {count}/{total} is officially in the books!",
        "Freshly baked: Slide {count}/{total} is complete."
    ],
    "validating_slides": [
        "Double-checking {count} slides for a polished look...",
        "Reviewing {count} slides for perfect alignment and fit...",
        "QA in progress: Inspecting {count} slides for any issues...",
        "Polishing the details on {count} slides...",
        "Ensuring everything is perfect across your {count} slides...",
        "Refining {count} slides for maximum impact...",
        "Performing a final quality check on {count} slides..."
    ],
    "validation_failed": [
        "Slide {idx}: Kept the original look (validation was tricky).",
        "Slide {idx}: No fixes needed, sticking with the original HTML.",
        "Slide {idx}: Validation was a bit complex, kept original version.",
        "Skipping polish for Slide {idx} to preserve your layout.",
        "Preserved original design for Slide {idx} ({done}/{total}).",
        "Slide {idx} looks good as is - no extra polishing applied.",
        "Decided to keep Slide {idx} in its natural state ({done}/{total})."
    ],
    "validation_fixed": [
        "Slide {idx} polished and perfected ({done}/{total})!",
        "Fixed it! Slide {idx} is now perfectly aligned ({done}/{total}).",
        "Success: Slide {idx} was validated and optimized ({done}/{total}).",
        "Done! Slide {idx} has been refined and polished ({done}/{total}).",
        "Good news! Slide {idx} is now pixel-perfect ({done}/{total}).",
        "Polished Slide {idx} to perfection ({done}/{total})!",
        "Success! Slide {idx} is now looking its absolute best ({done}/{total})."
    ],
    "validation_error": [
        "Encountered a hiccup on Slide {idx}, keeping original HTML.",
        "Slide {idx}: Error during validation, using original design.",
        "Keeping it safe: Original HTML used for Slide {idx} after an error.",
        "Skipping validation for Slide {idx} due to a minor internal error.",
        "Slide {idx} error ({done}/{total}) - used the initial draft.",
        "A small technical glitch on Slide {idx}, but original HTML is safe.",
        "Slide {idx} validation skipped to be safe - original look preserved."
    ],
    "missing_slides": [
        "Wait! Only detected {count}/{total} slides. Investigating...",
        "Heads up: Some slides ({count}/{total}) seem to be missing.",
        "Only found {count} out of {total} slides. Double-checking...",
        "Alert: {count}/{total} slides detected. Checking the pipeline...",
        "Hmm, {count}/{total} slides were found. Let me see what happened.",
        "Pipeline check: {count} out of {total} slides successfully made it.",
        "Only {count} slides found - checking if we missed anything!"
    ],
    "complete_success": [
        "Hooray! Your complete {count}-slide presentation is ready!",
        "Success! {count} beautiful slides have been generated for you.",
        "All done! Your {count}-slide deck is ready to shine.",
        "Mission accomplished: {count} slides are ready for your audience!",
        "Bravo! You've got {count} fresh slides ready to go!",
        "Your {count}-slide presentation is now a reality!",
        "Celebration time! {count} slides have been created flawlessly."
    ],
    "partial_success": [
        "All set! Generated {count}/{total} slides for you.",
        "Complete! {count}/{total} slides were successfully created.",
        "Done! Wrapped up with {count} out of {total} slides.",
        "Your presentation is ready with {count}/{total} slides.",
        "Presentation generated! {count}/{total} slides made it through.",
        "We've produced {count} out of {total} slides for your review.",
        "Session complete with {count}/{total} slides successfully built."
    ]
}


FRIENDLY_STATUS_MESSAGES = {
    "creating_outline": [
        "I'm mapping out the structure for your presentation...",
        "Just a moment, I'm crafting a custom outline for you...",
        "I'm brainstorming the best flow for your slides...",
        "I'm organizing your ideas into a clear plan...",
        "Designing the blueprint for your presentation now...",
        "I'm putting together a tailored flow just for you...",
        "Laying the foundation for your slides..."
    ],
    "outline_created": [
        "Great news! Your {total_slides}-slide outline is ready.",
        "All done with the outline! I've mapped out {total_slides} slides.",
        "Perfect! I've finalized a plan for {total_slides} slides.",
        "Your structure is set! We have {total_slides} slides ready to go.",
        "I've planned a nice {total_slides}-slide journey for you.",
        "Your presentation roadmap is ready with {total_slides} slides!",
        "Outline locked in! {total_slides} slides are ready for design."
    ],
    "generating_slides": [
        "Now I'm selecting templates and building your {total_slides} slides...",
        "I'm bringing your {total_slides} slides to life with great designs...",
        "Time to make it look good! Generating {total_slides} slides...",
        "I'm creating {total_slides} beautiful slides based on your plan...",
        "Customizing {total_slides} slides just for you...",
        "I'm picking the best layouts for your {total_slides} slides...",
        "Applying the finishing touches to your {total_slides} slides..."
    ],
    "slide_ready": [
        "Slide {idx}/{total} is ready for action.",
        "Done! Slide {idx}/{total} is prepped.",
        "Slide {idx}/{total} is looking good and ready to go.",
        "Setting up: Slide {idx}/{total} is all set.",
        "Ready! Slide {idx}/{total} is moving to the next step.",
        "Slide {idx}/{total} is ready for its transformation.",
        "All set for Slide {idx}/{total}!"
    ],
    "slide_fallback": [
        "Slide {idx}/{total} is getting a unique custom design.",
        "Slide {idx}/{total}: unique design (no template fit).",
        "I'm customizing Slide {idx}/{total} with a smart fallback design.",
        "Adjusting Slide {idx}/{total} to look unique.",
        "Slide {idx}/{total} is getting special custom treatment.",
        "No standard template for Slide {idx}/{total}, so I'm making it custom!",
        "Crafting Slide {idx}/{total} from scratch for a perfect fit."
    ],
    "slide_generated": [
        "I've just finished Slide {count}/{total}!",
        "Slide {count}/{total} is alive!",
        "Another one done! Slide {count}/{total} is ready.",
        "Moving along: Slide {count}/{total} is complete.",
        "Success: Slide {count}/{total} is now in your deck.",
        "Slide {count}/{total} is officially done!",
        "Freshly made: Slide {count}/{total} is complete."
    ],
    "validating_slides": [
        "I'm checking {count} slides to make sure they look great...",
        "Reviewing {count} slides for alignment and fit...",
        "Just checking {count} slides for any small issues...",
        "Polishing up the details on {count} slides...",
        "Making sure everything is perfect on your {count} slides...",
        "Refining {count} slides so they look their best...",
        "Doing a final quality check on {count} slides..."
    ],
    "validation_failed": [
        "Slide {idx}: Kept the original look.",
        "Slide {idx}: No changes needed, sticking with the original.",
        "Slide {idx}: Validation was tricky, so I kept the original.",
        "Skipping extra polish for Slide {idx} to keep your layout safe.",
        "Kept the original design for Slide {idx} ({done}/{total}).",
        "Slide {idx} looks good as is.",
        "I decided to keep Slide {idx} just as it was ({done}/{total})."
    ],
    "validation_fixed": [
        "Slide {idx} is polished and ready ({done}/{total})!",
        "Fixed it! Slide {idx} is now perfectly aligned ({done}/{total}).",
        "Success: Slide {idx} has been optimized ({done}/{total}).",
        "Done! Slide {idx} is refined and polished ({done}/{total}).",
        "Good news! Slide {idx} is looking perfect ({done}/{total}).",
        "I've polished Slide {idx} to perfection ({done}/{total})!",
        "Slide {idx} is now looking its absolute best ({done}/{total})."
    ],
    "validation_error": [
        "Had a small hiccup on Slide {idx}, keeping the original.",
        "Slide {idx}: Minor issue during check, using original design.",
        "Playing it safe: Original design used for Slide {idx}.",
        "Skipping check for Slide {idx} to be safe.",
        "Slide {idx} had a small glitch, so I used the draft ({done}/{total}).",
        "Technical glitch on Slide {idx}, but the slide is safe.",
        "Slide {idx} check skipped - original look preserved."
    ],
    "missing_slides": [
        "Hold on, I only see {count}/{total} slides. Checking...",
        "Heads up: Some slides ({count}/{total}) seem to be missing.",
        "I only found {count} out of {total} slides. Let me check.",
        "Alert: {count}/{total} slides detected. Checking the pipeline...",
        "Hmm, {count}/{total} slides found. Investigating...",
        "Pipeline check: {count} out of {total} slides made it.",
        "Only {count} slides found - checking if I missed anything!"
    ],
    "complete_success": [
        "Hooray! Your {count}-slide presentation is ready!",
        "Success! I've generated {count} beautiful slides for you.",
        "All done! Your {count}-slide deck is ready.",
        "Mission accomplished: {count} slides are ready for you!",
        "You've got {count} fresh slides ready to go!",
        "Your {count}-slide presentation is real!",
        "Celebration time! {count} slides created successfully."
    ],
    "partial_success": [
        "All set! I made {count}/{total} slides for you.",
        "Complete! {count}/{total} slides were successfully created.",
        "Done! I finished {count} out of {total} slides.",
        "Your presentation is ready with {count}/{total} slides.",
        "Presentation generated! {count}/{total} slides are here.",
        "I've produced {count} out of {total} slides for you.",
        "Session complete with {count}/{total} slides."
    ]
}

def get_status_msg(key, style="friendly", **kwargs):
    """
    Select a random message for a given status key and format it.
    
    Args:
        key (str): The status key to look up.
        style (str): The style of message ('default' or 'friendly').
        **kwargs: Arguments for string formatting.
    """
    messages = STATUS_MESSAGES
    if style == "friendly":
        messages = FRIENDLY_STATUS_MESSAGES
        
    if key not in messages:
        return f"Unknown status: {key}"
    template = random.choice(messages[key])
    return template.format(**kwargs)
